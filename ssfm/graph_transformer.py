import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.flow_map import AbstractEMStepModel
from ssfm.typing import Y


def _time_features(s: Float[Array, ""], t: Float[Array, ""]) -> Float[Array, "8"]:
    return jnp.stack(
        [
            s - 0.5,
            jnp.cos(2 * jnp.pi * s),
            jnp.sin(2 * jnp.pi * s),
            -jnp.cos(4 * jnp.pi * s),
            t - 0.5,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t),
        ]
    )


class TimeEmbedding(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, time_dim: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(8, time_dim, key=k1)
        self.linear2 = eqx.nn.Linear(time_dim, time_dim, key=k2)

    def __call__(
        self, s: Float[Array, ""], t: Float[Array, ""]
    ) -> Float[Array, "time_dim"]:
        h = _time_features(s, t)
        h = jax.nn.silu(self.linear1(h))
        return self.linear2(h)


class GatedResidual(eqx.Module):
    gate: eqx.nn.Linear

    def __init__(self, hidden_nf: int, *, key: PRNGKeyArray):
        self.gate = eqx.nn.Linear(3 * hidden_nf, 1, use_bias=False, key=key)

    def __call__(
        self,
        x: Float[Array, "n hidden"],
        res: Float[Array, "n hidden"],
    ) -> Float[Array, "n hidden"]:
        gate_in = jnp.concatenate([x, res, x - res], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.gate)(gate_in))
        return x * gate + res * (1 - gate)


class Attention(eqx.Module):
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear
    to_edge: eqx.nn.Linear
    to_out: eqx.nn.Linear
    heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_nf: int,
        heads: int = 8,
        dim_head: int = 64,
        *,
        key: PRNGKeyArray,
    ):
        k_q, k_k, k_v, k_e, k_o = jax.random.split(key, 5)
        inner_dim = heads * dim_head
        self.to_q = eqx.nn.Linear(hidden_nf, inner_dim, key=k_q)
        self.to_k = eqx.nn.Linear(hidden_nf, inner_dim, key=k_k)
        self.to_v = eqx.nn.Linear(hidden_nf, inner_dim, key=k_v)
        self.to_edge = eqx.nn.Linear(hidden_nf, inner_dim, key=k_e)
        self.to_out = eqx.nn.Linear(inner_dim, hidden_nf, key=k_o)
        self.heads = heads
        self.dim_head = dim_head

    def __call__(
        self,
        nodes: Float[Array, "n hidden"],
        edges: Float[Array, "n n hidden"],
    ) -> Float[Array, "n hidden"]:
        h, d = self.heads, self.dim_head
        n = nodes.shape[0]
        scale = d**-0.5

        q = jax.vmap(self.to_q)(nodes).reshape(n, h, d)
        k = jax.vmap(self.to_k)(nodes).reshape(n, h, d)
        v = jax.vmap(self.to_v)(nodes).reshape(n, h, d)
        e_kv = jax.vmap(jax.vmap(self.to_edge))(edges).reshape(n, n, h, d)

        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))
        e_kv = jnp.transpose(e_kv, (2, 0, 1, 3))

        k = k[:, None, :, :] + e_kv
        v = v[:, None, :, :] + e_kv

        sim = jnp.einsum("hid,hijd->hij", q, k) * scale
        attn = jax.nn.softmax(sim, axis=-1)
        out = jnp.einsum("hij,hijd->hid", attn, v)

        out = jnp.transpose(out, (1, 0, 2)).reshape(n, h * d)
        return jax.vmap(self.to_out)(out)


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_nf: int,
        ff_mult: int = 4,
        dropout_rate: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(hidden_nf, ff_mult * hidden_nf, key=k1)
        self.linear2 = eqx.nn.Linear(ff_mult * hidden_nf, hidden_nf, key=k2)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(
        self,
        h: Float[Array, "n hidden"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "n hidden"]:
        h = jax.vmap(self.linear1)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.linear2)(h)
        return self.dropout(h, key=key)


class TransformerBlock(eqx.Module):
    attn_norm: eqx.nn.LayerNorm
    attn: Attention
    attn_gate: GatedResidual
    ff_norm: eqx.nn.LayerNorm
    ff: FeedForward
    ff_gate: GatedResidual

    def __init__(
        self,
        hidden_nf: int,
        heads: int,
        dim_head: int,
        ff_mult: int,
        dropout_rate: float,
        *,
        key: PRNGKeyArray,
    ):
        k_attn, k_ag, k_ff, k_fg = jax.random.split(key, 4)
        self.attn_norm = eqx.nn.LayerNorm((hidden_nf,))
        self.attn = Attention(hidden_nf, heads, dim_head, key=k_attn)
        self.attn_gate = GatedResidual(hidden_nf, key=k_ag)
        self.ff_norm = eqx.nn.LayerNorm((hidden_nf,))
        self.ff = FeedForward(hidden_nf, ff_mult, dropout_rate, key=k_ff)
        self.ff_gate = GatedResidual(hidden_nf, key=k_fg)

    def __call__(
        self,
        nodes: Float[Array, "n hidden"],
        edges: Float[Array, "n n hidden"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "n hidden"]:
        a = self.attn(jax.vmap(self.attn_norm)(nodes), edges)
        nodes = self.attn_gate(a, nodes)
        f = self.ff(jax.vmap(self.ff_norm)(nodes), key=key)
        nodes = self.ff_gate(f, nodes)
        return nodes


class GraphTransformerBackbone(eqx.Module):
    node_embed: eqx.nn.Linear
    edge_embed: eqx.nn.Linear
    blocks: list[TransformerBlock]
    decoder: eqx.nn.Linear
    n_atoms: int = eqx.field(static=True)
    hidden_nf: int = eqx.field(static=True)
    use_edges: bool = eqx.field(static=True)

    def __init__(
        self,
        n_atoms: int,
        node_in_dim: int,
        hidden_nf: int,
        n_layers: int,
        heads: int,
        dim_head: int,
        ff_mult: int,
        dropout_rate: float,
        use_edges: bool,
        *,
        key: PRNGKeyArray,
    ):
        k_node, k_edge, k_dec, *k_blocks = jax.random.split(key, 3 + n_layers)
        self.node_embed = eqx.nn.Linear(n_atoms + node_in_dim, hidden_nf, key=k_node)
        self.edge_embed = eqx.nn.Linear(3, hidden_nf, key=k_edge)
        self.blocks = [
            TransformerBlock(hidden_nf, heads, dim_head, ff_mult, dropout_rate, key=kb)
            for kb in k_blocks
        ]
        self.decoder = eqx.nn.Linear(n_atoms * hidden_nf, n_atoms * 3, key=k_dec)
        self.n_atoms = n_atoms
        self.hidden_nf = hidden_nf
        self.use_edges = use_edges

    def __call__(
        self,
        coords: Float[Array, "n 3"] | None,
        node_features: Float[Array, "n F"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "n 3"]:
        atom_id = jnp.eye(self.n_atoms)
        h = jnp.concatenate([atom_id, node_features], axis=-1)
        h = jax.vmap(self.node_embed)(h)

        if self.use_edges:
            assert coords is not None, "use_edges=True requires coords"
            diff = coords[:, None, :] - coords[None, :, :]
            e = jax.vmap(jax.vmap(self.edge_embed))(diff)
        else:
            e = jnp.zeros((self.n_atoms, self.n_atoms, self.hidden_nf))

        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            h = block(h, e, key=subkey)

        out = self.decoder(h.reshape(-1))
        return out.reshape(self.n_atoms, 3)


class GraphTransformerEMStepModel(AbstractEMStepModel):
    drift_net: GraphTransformerBackbone
    diffusion_net: GraphTransformerBackbone
    time_embed: TimeEmbedding
    n_atoms: int = eqx.field(static=True)

    def __init__(
        self,
        n_atoms: int,
        hidden_nf: int = 96,
        n_layers: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        time_dim: int = 64,
        dropout_rate: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        k_time, k_drift, k_diff = jax.random.split(key, 3)
        self.time_embed = TimeEmbedding(time_dim, key=k_time)
        node_in_dim = time_dim + 9
        self.drift_net = GraphTransformerBackbone(
            n_atoms=n_atoms,
            node_in_dim=node_in_dim,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout_rate=dropout_rate,
            use_edges=True,
            key=k_drift,
        )
        self.diffusion_net = GraphTransformerBackbone(
            n_atoms=n_atoms,
            node_in_dim=node_in_dim,
            hidden_nf=hidden_nf,
            n_layers=n_layers - 1,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout_rate=dropout_rate,
            use_edges=False,
            key=k_diff,
        )
        self.n_atoms = n_atoms

    def _node_features(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
    ) -> Float[Array, "n F"]:
        tau = self.time_embed(s, t)
        tau_tiled = jnp.tile(tau[None, :], (self.n_atoms, 1))
        W3 = W.reshape(self.n_atoms, 3)
        H3 = H.reshape(self.n_atoms, 3)
        K3 = K.reshape(self.n_atoms, 3)
        return jnp.concatenate([tau_tiled, W3, H3, K3], axis=-1)

    def drift(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        coords = ys.reshape(self.n_atoms, 3)
        node_feats = self._node_features(s, t, W, H, K)
        out = self.drift_net(coords, node_feats, key=key)
        return out.reshape(-1)

    def diffusion(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        node_feats = self._node_features(s, t, W, H, K)
        out = self.diffusion_net(None, node_feats, key=key)
        return out.reshape(-1)
