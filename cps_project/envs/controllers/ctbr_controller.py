import torch


class CTRBctrl:
    def __init__(self, num_envs) -> None:
        self.body_drone_angvels = torch.zeros(
            (num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.body_drone_linvels = torch.zeros_like(self.body_drone_angvels)
        self.error = torch.zeros_like(self.body_drone_angvels)
        self.tau_des = torch.zeros_like(self.body_drone_angvels)

        self.B = torch.zeros((num_envs, 4), device=self.device, dtype=torch.float32)
        self.thrust = torch.zeros_like(self.B)

        # Drone parameters
        # equations ref https://rpg.ifi.uzh.ch/docs/ICRA15_Faessler.pdf
        self.real_thrust_upbound = torch.tensor([0.15], device=self.device)
        self.real_thrust_lowbound = torch.tensor([0.0], device=self.device)
        diag = 0.04
        rot_tau_coeff = 0.00596
        # Parameters matrix of the drone
        self.base_A = torch.tensor(
            [
                [diag, -diag, -diag, diag],
                [-diag, -diag, diag, diag],
                [rot_tau_coeff, -rot_tau_coeff, rot_tau_coeff, -rot_tau_coeff],
                [1, 1, 1, 1],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.A = self.base_A.repeat(num_envs, 1, 1)
        self.inv_A = torch.linalg.inv(self.A)

        # Proportional gain matrix
        self.P = torch.zeros(
            (self.num_envs, 3, 3), device=self.device, dtype=torch.float32
        )
        self.P[:, 0, 0] = 50
        self.P[:, 1, 1] = 50
        self.P[:, 2, 2] = 50

        # Inertia matrix
        self.J = torch.zeros(
            (self.num_envs, 3, 3), device=self.device, dtype=torch.float32
        )
        self.J[:, 0, 0] = 2.3951e-5
        self.J[:, 1, 1] = 2.3951e-5
        self.J[:, 2, 2] = 3.2347e-5

    @torch.no_grad()
    def update(self, actions, drone_quats, drone_angvels, drone_linvels):
        """
        Implements the P + FB linearizing controller to track the desired CTBR actions
        """

        # Compute velocities wrt drone body frame
        self.body_drone_angvels[:] = self.quat_rotate_inverse(
            drone_quats, drone_angvels
        )
        self.body_drone_linvels[:] = self.quat_rotate_inverse(
            drone_quats, drone_linvels
        )

        # Compute error term
        self.error[:] = actions[:, 0:3] - self.body_drone_angvels

        ### PX4 controller: Compute the control action
        ## Proportional term
        prop_temp = torch.einsum("nhj,nj-> nh", self.P, self.error)
        prop = torch.einsum("nhj,nj-> nh", self.J, prop_temp)

        ## Feedback linearization term
        fb_lin = torch.zeros_like(prop)
        fb_lin[:] = torch.cross(
            self.body_drone_angvels,
            torch.bmm(
                self.J, torch.unsqueeze(self.body_drone_angvels, dim=2)
            ).squeeze(),
            dim=1,
        )

        ## Overall control action
        self.tau_des[:] = prop + fb_lin

        # Solve the linear system (thrust mixing) to find actual tagert force on each rotor
        self.B[..., 0:3] = self.tau_des
        self.B[..., 3] = ((actions[..., 3] + 1) / 2) * (self.real_thrust_upbound * 4)

        # Now compute rotor force solving the linear sistem of eq rotor_forces[num_envs, 4]
        # Solve system of lin equations: AX = B, obtain 4 scalars that are the value of the thrusts of each motor
        # Thrust: vector perpendicular to the drone plane applied at the center of the motor
        self.thrust[:] = torch.linalg.solve(self.A, self.B)

        # PX4: Saturated mixing, Airmode Enabled
        # Check if some motors saturates and rescale to avoid this condition
        thrust_offset_lower = torch.min(self.thrust, dim=-1)[0]
        thrust_offset_idx = torch.where(thrust_offset_lower < 0.0)[0]
        if len(thrust_offset_idx) > 0:
            self.thrust[thrust_offset_idx] -= thrust_offset_lower[
                thrust_offset_idx, None
            ].repeat(1, self.thrust.shape[1])

        thrust_offset_upper = (
            torch.max(self.thrust, dim=-1)[0] - self.real_thrust_upbound
        )
        thrust_offset_idx2 = torch.where(thrust_offset_upper > 0.0)[0]
        if len(thrust_offset_idx2) > 0:
            self.thrust[thrust_offset_idx2] -= thrust_offset_upper[
                thrust_offset_idx2, None
            ].repeat(1, self.thrust.shape[1])

        # Clamp at the end if not inside the feasible thrust range
        self.thrust[:] = torch.clamp(
            self.thrust, self.real_thrust_lowbound, self.real_thrust_upbound
        )

        # Remap individual rotor thrusts to the resulting linear force and moment to be compatible with Isaac
        self.B[:] = torch.linalg.solve(self.inv_A, self.thrust)

        total_torque = self.B[:, 0:3].clone()
        common_thrust = self.B[:, 3].clone()

        return total_torque, common_thrust

    @torch.jit.script
    def quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = (
            q_vec
            * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
            * 2.0
        )


# if __name__ == "__main__":
## EXAMPLE - PSEUDO CODE: not working!!

# sim_steps = 1000
# num_envs = 2048
# bodies_per_env = 1
# device = "gpu"
# controller = CTRBctrl(num_envs)
# sim = Isaacgym.create_simulation()
# nn = Model.mlp()
# friction = torch.zeros(
#     (num_envs, bodies_per_env, 3), device=device, dtype=torch.float32
# )

# for _ in range(sim_steps):
#     state = sim.step()
#     actions = nn(state)

#     total_torque, common_thrust = controller.update(
#         actions, state.drone_quats, state.drone_angvels, state.drone_linvels
#     )

#     # Compute Friction forces (opposite to drone vels)
#     friction[:, drone_handle, :] = (
#         -0.02 * torch.sign(body_drone_linvels) * body_drone_linvels**2
#     )
#     tot_f = common_thrust + friction

#     # Apply forces and torques to the drone
#     gym.apply_rigid_body_force_tensors(
#         sim,
#         gymtorch.unwrap_tensor(tot_f),
#         gymtorch.unwrap_tensor(total_torque),
#         gymapi.LOCAL_SPACE,
#     )

# print("Done")
