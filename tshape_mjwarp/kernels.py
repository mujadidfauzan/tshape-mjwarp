"""Contact reductions on device; inactive capacity slots never enter diagnostics."""
import torch


def reduce_contacts(geom, world, distance, address, count, num_envs, num_geoms,
                    agent_map, object_map, obstacle_map):
    valid = (torch.arange(len(world), device=world.device) < count[0]) & (address[:, 0] >= 0)
    w = world.long().clamp(0, num_envs-1)
    g = geom.long().clamp(0, num_geoms-1)
    penetration = torch.zeros(num_envs, device=world.device)
    penetration.scatter_reduce_(0, w, torch.where(valid, (-distance).clamp_min(0), 0.), reduce="amax")
    # Caller slices the allocated geom-width contact flags to the actual agent count.
    flags = torch.zeros((num_envs, num_geoms), device=world.device, dtype=torch.int32)
    for side in (0, 1):
        agent = agent_map[g[:, side]]
        match = valid & (agent >= 0) & object_map[g[:, 1-side]]
        flags.flatten().scatter_reduce_(0, w*num_geoms+agent.clamp_min(0), match.int(), reduce="amax")
    pair = torch.zeros((num_envs, num_geoms*num_geoms), device=world.device, dtype=torch.int32)
    obstacle = valid & (obstacle_map[g[:, 0]] | obstacle_map[g[:, 1]])
    pair.flatten().scatter_reduce_(0, w*num_geoms*num_geoms+g.amin(1)*num_geoms+g.amax(1),
                                  obstacle.int(), reduce="amax")
    return flags.bool(), penetration, pair.bool()
