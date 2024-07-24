import torch
import functools
import operator
import einops

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


def inter_residue_violation_loss(
    pos14: Tensor,
    pos14_mask: Tensor,
    seq: Tensor,
    aa: Tensor,
    tolerance_factor: float,
) -> tuple[Tensor, Tensor]:
    """Compute inter residue bond loss, i.e.,
    all the losses between pairs of atoms connected by a chemical bond yet not within the same residue.

    See alphafold supplementary 1.9.11 for details.

    Args:
        pos14: Atom position in pos14 representation, (B, N, 14, 3).
        pos14_mask: Atom mask in pos14 representation, (B, N, 14).
        aa: Amino acid type, (B, N).
        seq: Residue index, (B, N).
        tolerance_factor: Tolerance factor measured in standard deviations of pdb distributions.

    Returns:
        loss: sum of c_n_loss/ca_c_n_loss/c_n_ca_loss, (B).
        error: sum of c_n_error/ca_c_n_error/c_n_ca_error, (B, N-1).
    """
    c_pos = pos14[..., :-1, protein_constants.ATOM_C, :]
    c_mask = pos14_mask[..., :-1, protein_constants.ATOM_C]
    ca_pos = pos14[..., :-1, protein_constants.ATOM_CA, :]
    ca_mask = pos14_mask[..., :-1, protein_constants.ATOM_CA]
    next_n_pos = pos14[..., 1:, protein_constants.ATOM_N, :]
    next_n_mask = pos14_mask[..., 1:, protein_constants.ATOM_N]
    next_ca_pos = pos14[..., 1:, protein_constants.ATOM_CA, :]
    next_ca_mask = pos14_mask[..., 1:, protein_constants.ATOM_CA]
    non_gap_mask = (seq[..., -1:] - seq[..., 1:]) == 1.0

    c_n_bond_length = torch.linalg.vector_norm(c_pos - next_n_pos, dim=-1)
    next_proline_mask = torch.eq(aa[..., 1:], protein_constants.resname_three_to_index["PRO"])
    gt: Tensor | float = torch.index_select(
        torch.tensor(protein_constants.INTER_RES_BOUND_LENGTH_C_N, device=c_pos.device),
        0,
        next_proline_mask.long().view(-1),
    ).view(next_proline_mask.shape)
    gt_stddev: Tensor | float = torch.index_select(
        torch.tensor(protein_constants.INTER_RES_BOUND_LENGTH_STDDEV_C_N, device=c_pos.device),
        0,
        next_proline_mask.long().view(-1),
    ).view(next_proline_mask.shape)
    c_n_bond_length_error = torch.abs(gt - c_n_bond_length)
    c_n_loss = F.relu(c_n_bond_length_error - tolerance_factor * gt_stddev)
    c_n_mask = non_gap_mask * c_mask * next_n_mask
    c_n_loss, c_n_error = common.mask.masked_mean(c_n_loss, c_n_mask, return_masked=True)

    c_ca_unit_vec = common.geometric.normalize_vector(ca_pos - c_pos, -1)
    c_n_unit_vec = common.geometric.normalize_vector(next_n_pos - c_pos, -1)
    n_ca_unit_vec = common.geometric.normalize_vector(next_ca_pos - next_n_pos, -1)

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, -1)
    gt = protein_constants.INTER_RES_COS_ANGLES_CA_C_N[0]
    gt_stddev = protein_constants.INTER_RES_COS_ANGLES_CA_C_N[1]
    ca_c_n_cos_angle_error = torch.abs(gt - ca_c_n_cos_angle)
    ca_c_n_loss = F.relu(ca_c_n_cos_angle_error - tolerance_factor * gt_stddev)
    ca_c_n_mask = non_gap_mask * ca_mask * c_mask * next_n_mask
    ca_c_n_loss, ca_c_n_error = common.mask.masked_mean(ca_c_n_loss, ca_c_n_mask, return_masked=True)

    c_n_ca_cos_angle = torch.sum(-c_n_unit_vec * n_ca_unit_vec, -1)
    gt = protein_constants.INTER_RES_COS_ANGLES_C_N_CA[0]
    gt_stddev = protein_constants.INTER_RES_COS_ANGLES_C_N_CA[1]
    c_n_ca_cos_angle_error = torch.abs(gt - c_n_ca_cos_angle)
    c_n_ca_loss = F.relu(c_n_ca_cos_angle_error - tolerance_factor * gt_stddev)
    c_n_ca_mask = non_gap_mask * c_mask * next_n_mask * next_ca_mask
    c_n_ca_loss, c_n_ca_error = common.mask.masked_mean(c_n_ca_loss, c_n_ca_mask, return_masked=True)

    loss = c_n_loss + ca_c_n_loss + c_n_ca_loss
    error = c_n_error + ca_c_n_error + c_n_ca_error
    return loss, error


def clash_loss(
    pos14: Tensor,
    pos14_mask: Tensor,
    seq: Tensor,
    aa: Tensor,
    clash_tolerance: float,
    average_over_clash: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute inter residue clash loss.

    Clash loss is confusing because different sources give different implementation.

                                    monomer                     multimer
    For every pair of atoms i, j that exists:
    According to paper:             ∑_ij(clash_ij)              Average_{ij s.t. clash_ij>0}(clash_ij)
    According to JAX code:          Average_i(∑_j(clash_ij))    Average_i(∑_j(clash_ij))
    According to OpenFold code:     Average_i(∑_j(clash_ij))    Average_i(Average_{j s.t. clash_ij>0}(clash_ij))

    Here we use the following implementation:
    FoldY:                          Average_i(∑_j(clash_ij))    Average_{j s.t. clash_ij>0}(clash_ij)
    average_over_clash:             False                       True

    See alphafold supplementary 1.9.11 & multimer 2.5 for details.

    Args:
        pos14: Atom position in pos14 representation, (B, N, 14, 3).
        pos14_mask: Atom mask in pos14 representation, (B, N, 14).
        aa: Amino acid type, (B, N).
        seq: Residue index, (B, N).
        clash_tolerance: Clash distance tolerance.
        average_over_clash:
            False(monomer): Average over all atoms, sum over clash for one atom.
            True(multimer): Average over clash pairs.

    Returns:
        loss: clash loss, (B).
        error: per-residue-pair clash error, (B, N, N).
    """
    dist = common.geometric.compute_atom_pair_dist(pos14)

    c_n_bond_mask = F.pad(
        torch.eq(
            einops.repeat(seq, "... i -> ... i j n c", j=1, c=1, n=1) + 1,
            einops.repeat(seq, "... j -> ... i j n c", i=1, c=1, n=1),
        ),
        (
            protein_constants.ATOM_N,
            13 - protein_constants.ATOM_N,
            protein_constants.ATOM_C,
            13 - protein_constants.ATOM_C,
        ),
    )
    cys_mask = torch.eq(aa, protein_constants.resname_three_to_index["CYS"])
    atom_sg = protein_constants.RESNAME_TO_ATOM14_NAMES["CYS"].index("SG")
    disulfide_bond_mask = F.pad(
        einops.rearrange(common.mask.node_mask_to_pair_mask(cys_mask), "... -> ... 1 1"),
        (atom_sg, 13 - atom_sg, atom_sg, 13 - atom_sg),
    )
    inter_chain_pair_mask = torch.logical_not(c_n_bond_mask) * torch.logical_not(disulfide_bond_mask)
    # that is applied to all i, j atom pairs that exist,
    # but C-N peptide bonds in backbone and S-S disulfide bonds are masked
    intra_chain_bond_mask = torch.index_select(
        protein_constants.restype_atom14_pair_to_bond_mask.to(aa.device), 0, torch.clamp(aa, max=20).view(-1)
    ).view(*aa.size(), 14, 14)
    intra_chain_bond_mask = einops.repeat(intra_chain_bond_mask, "... i n c -> ... i j n c", j=1)
    intra_chain_pair_mask = torch.logical_not(intra_chain_bond_mask)

    intra_chain_mask = torch.eq(
        einops.repeat(seq, "... i -> ... i j n c", j=1, n=1, c=1),
        einops.repeat(seq, "... j -> ... i j n c", i=1, n=1, c=1),
    )

    atom_seq = einops.repeat(seq, "... i -> ... i n", n=1) * 14 + torch.arange(14, device=aa.device).view(
        *[1] * len(seq.shape), 14
    )
    sequential_mask = torch.lt(
        einops.repeat(atom_seq, "... i n -> ... i j n c", j=1, c=1),
        einops.repeat(atom_seq, "... j c -> ... i j n c", i=1, n=1),
    )
    pair_mask = (
        sequential_mask
        * torch.where(intra_chain_mask, intra_chain_pair_mask, inter_chain_pair_mask)
        * torch.logical_or(inter_chain_pair_mask, intra_chain_pair_mask)
        * einops.repeat(pos14_mask, "... i n_atoms_i -> ... i j n_atoms_i n_atoms_j", j=1, n_atoms_j=1)
        * einops.repeat(pos14_mask, "... j n_atoms_j -> ... i j n_atoms_i n_atoms_j", i=1, n_atoms_i=1)
    )

    pos14_vdw_radius = torch.index_select(
        protein_constants.restype_to_atom14_vdw_radius.to(aa.device), 0, torch.clamp(aa, max=20).view(-1)
    ).view(pos14_mask.shape)
    dist_lower_bound = einops.repeat(
        pos14_vdw_radius, "... i n_atoms_i -> ... i j n_atoms_i n_atoms_j", j=1, n_atoms_j=1
    ) + einops.repeat(pos14_vdw_radius, "... j n_atoms_j -> ... i j n_atoms_i n_atoms_j", i=1, n_atoms_i=1)
    clash_error = F.relu(dist_lower_bound - dist - clash_tolerance) * pair_mask  # [..., N, N, 14, 14]

    if average_over_clash:
        loss = common.mask.masked_mean(clash_error, clash_error > 0, (-1, -2, -3, -4))
    else:
        per_atom_clash_error = clash_error.sum((-1, -3)) + clash_error.sum((-2, -4))
        loss = common.mask.masked_mean(per_atom_clash_error, pos14_mask, (-1, -2))
    error = clash_error.sum((-1, -2))
    return loss, error


def intra_residue_bond_violation_loss(
    pos14: Tensor,
    pos14_mask: Tensor,
    aa: Tensor,
    tolerance_factor: float,
) -> tuple[Tensor, Tensor]:
    """Compute intra residue bond loss.

    It must return 0 for sanity checking, because all the intra-residue
    bonds are generated from T_bb & torsions with fixed bond lengths.

    See alphafold supplementary 1.9.11 for details.

    Args:
        pos14: Atom position in pos14 representation, (B, N, 14, 3).
        pos14_mask: Atom mask in pos14 representation, (B, N, 14).
        aa: Amino acid type, (B, N).
        tolerance_factor: Tolerance factor measured in standard deviations of pdb distributions.

    Returns:
        loss: Intra-residue bond length loss, (B).
        error: Intra-residue bond length per-residue error, (B, N).
    """
    dist = common.geometric.cdist(pos14, pos14)

    sequential_mask = torch.lt(
        einops.repeat(torch.arange(14, device=aa.device), "i -> i j", j=1),
        einops.repeat(torch.arange(14, device=aa.device), "j -> i j", i=1),
    )
    bond_mask = torch.index_select(
        protein_constants.restype_atom14_pair_to_bond_mask.to(aa.device), 0, aa.view(-1)
    ).view(*aa.size(), 14, 14)
    pair_mask = sequential_mask * bond_mask * tensor_utils.node2pair(pos14_mask, pos14_mask, -1, torch.logical_and)

    gt = torch.index_select(protein_constants.restype_atom14_pair_to_bond_length.to(aa.device), 0, aa.view(-1)).view(
        *aa.size(), 14, 14
    )
    gt_stddev = torch.index_select(
        protein_constants.restype_atom14_pair_to_bond_length_stddev.to(aa.device),
        0,
        aa.view(-1),
    ).view(*aa.size(), 14, 14)
    bond_error = (dist - gt).abs()
    bond_loss = F.relu(bond_error - tolerance_factor * gt_stddev)  # (B, N, 14, 14)

    loss = common.mask.masked_mean(bond_loss, pair_mask, (-1, -2, -3))
    error = bond_loss.sum((-1, -2))
    return loss, error


def find_structural_violation(
    pos14: Tensor,
    pos14_mask: Tensor,
    seq: Tensor,
    aa: Tensor,
    tolerance_factor: float,
    clash_tolerance: float,
    average_over_clash: bool = False,
) -> tuple[base.ProteinFoldingLoss, base.ProteinFoldingError]:
    """Compute the structural violation loss.

    The structural violation loss comes from three types of violations:
    a. Bond length loss between any pair of bonding atoms.
    b. Bond angle loss applied on all bond angles. In practice, however,
       all the intra-residue bonds are constructed from T_bb & torsions with fixed bond angles,
       so this loss only comes from inter-residue bond angles, i.e.,
       ca_c_n loss & c_n_ca loss.
    c. Non-bond clash loss between any pair of non-bond atoms.

    However, the structural violation loss are calculated in three parts for convenience:
    1. inter_residue_bond_loss (including a & b)
    2. clash loss (including c)
    3. intra_residue_bond_loss (must be 0)

    See alphafold supplementary 1.9.11 for details.

    Args:
        pos14: Atom position in pos14 representation, (B, N, 14, 3).
        pos14_mask: Atom mask in pos14 representation, (B, N, 14).
        seq: Residue index, (B, N).
        aa: Amino acid type, (B, N).
        clash_tolerance: Clash distance tolerance.
        tolerance_factor: Tolerance factor measured in standard deviations of pdb distributions.
        average_over_clash: Bool. See clash_loss for details.

    Returns:
        losses: A named dict of:
                inter-residue violation loss, (B).
                clash loss, (B).
                intra-residue violation loss, (B).

        errors: A named dict of:
                inter-residue violation error, (B, N-1).
                clash error, (B, N, N).
                intra-residue violation error, (B, N).
    """
    losses: base.ProteinFoldingLoss = {}
    errors: base.ProteinFoldingError = {}

    l, e = inter_residue_violation_loss(pos14, pos14_mask, seq, aa, tolerance_factor)
    losses["inter_residue_violation"] = l
    errors["inter_residue_violation"] = e

    l, e = clash_loss(pos14, pos14_mask, seq, aa, clash_tolerance, average_over_clash)
    losses["clash"] = l
    errors["clash"] = e

    l, e = intra_residue_bond_violation_loss(pos14, pos14_mask, aa, tolerance_factor)
    losses["intra_residue_violation"] = l
    errors["intra_residue_violation"] = e

    return losses, errors


@tensor_utils.compute_in_fp32
def structural_violation_loss(
    pos14: Tensor,
    pos14_mask: Tensor,
    seq: Tensor,
    aa: Tensor,
    tolerance_factor: float,
    clash_tolerance: float,
    average_over_clash: bool = False,
) -> Tensor:
    """Compute the structural violation loss.

    See alphafold supplementary 1.9.11 for details.

    Args:
        pos14: Atom position in pos14 representation, (B, N, 14, 3).
        pos14_mask: Atom mask in pos14 representation, (B, N, 14).
        seq: Residue index, (B, N).
        aa: Amino acid type, (B, N).
        clash_tolerance: Clash distance tolerance.
        tolerance_factor: Tolerance factor measured in standard deviations of pdb distributions.
        average_over_clash: Bool. See clash_loss for details.

    Returns:
        structure violation loss, (B), which is the sum of inter-residue bond loss, inter-residue clash loss and
        intra-residue violation loss.
    """
    violations, _ = find_structural_violation(
        pos14, pos14_mask, seq, aa, tolerance_factor, clash_tolerance, average_over_clash
    )
    return functools.reduce(operator.add, violations.values())