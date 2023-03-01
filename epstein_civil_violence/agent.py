import math
from this import d, s

import mesa
import networkx as nx


class Citizen(mesa.Agent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        hardship,
        regime_legitimacy,
        risk_aversion,
        threshold,
        vision,
    ):
        """
        Create a new Citizen.
        Args:
            unique_id: unique int
            x, y: Grid coordinates
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(unique_id, model)
        self.breed = "citizen"
        self.pos = pos
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.condition = "Quiescent"
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None
        self.strong_ties_received = 0
        self.weak_ties_received = 0

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.
        
        if self.model.activation_type == "With SWN":

            self.prepare_ego()
            self.update_neighbors()
          #  self.update_estimated_arrest_probability()
            self.update_network_arrest_probability()
            net_risk = self.risk_aversion * self.arrest_probability
        
            # New update mechanism with probability pdfs
            if self.condition == "Active": 
                for edge in self.ego_network.edges(self.unique_id):
                    u, v = edge
                    u, v = (u, v) if u < v else (v, u)
                    self.model.link_to_censor.add((u, v))


            if self.condition == "Quiescent":
                if self.grievance - net_risk > self.threshold:
                    self.condition = "Active"

        
            if self.model.movement and self.empty_neighbors:
                new_pos = self.random.choice(self.empty_neighbors)
                self.model.grid.move_agent(self, new_pos)

        else:
            self.update_neighbors()
            self.update_estimated_arrest_probability()
            net_risk = self.risk_aversion * self.arrest_probability

            if self.condition == "Quiescent":
                if self.grievance - net_risk > self.threshold:
                    self.condition = "Active"

            if self.model.movement and self.empty_neighbors:
                new_pos = self.random.choice(self.empty_neighbors)
                self.model.grid.move_agent(self, new_pos)
    
    def get_agents_by_ids(self, id_list):
        return [self.model.agent_dict[i] for i in id_list if i in self.model.agent_dict]

    def prepare_ego(self):
        self.ego_network = nx.ego_graph(self.model.G, self.unique_id, radius = 2)
        ebunch = []
        for u,v,data in self.ego_network.edges.data():
            if data['censor_steps'] > 0:
                ebunch.append((u,v))
        self.ego_network.remove_edges_from(ebunch)
        self.ego_network = nx.ego_graph(self.ego_network, self.unique_id, radius=2)
        degree_1_neighbors = frozenset(self.ego_network.neighbors(self.unique_id))
        self.degree_1_neighbors = self.get_agents_by_ids(degree_1_neighbors)
        degree_2_neighbors = frozenset(self.ego_network.nodes()) - degree_1_neighbors
        self.degree_2_neighbors = self.get_agents_by_ids(degree_2_neighbors)
     #   print(degree_2_neighbors)

    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]

    def update_network_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood + the network neighbors
        , estimate the p(Arrest | I go active).
        """
        cops_in_vision = len([c for c in self.neighbors if c.breed == "cop"])
        actives_in_vision = 1.0  # citizen counts herself
        self.strong_ties_received = 0
        self.weak_ties_received = 0
        for c in self.neighbors:
            if (
                c.breed == "citizen"
                and c.condition == "Active"
                and c.jail_sentence == 0
            ):
                actives_in_vision += 1
        # Update SWN effects
        for c in self.degree_1_neighbors:
            if (
                c.condition == "Active"
                and c.jail_sentence == 0
            ):
                actives_in_vision += self.model.strong_weak_ratio
                self.strong_ties_received += 1

        for c in self.degree_2_neighbors:
            if (
                c.condition == "Active"
                and c.jail_sentence == 0
            ):
                actives_in_vision += 1
                self.weak_ties_received += 1

        self.actives_in_vision = actives_in_vision
        self.arrest_probability = 1 - math.exp(
            -1
            * self.model.arrest_prob_constant
            * math.floor(cops_in_vision / actives_in_vision)
        )

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = len([c for c in self.neighbors if c.breed == "cop"])
        actives_in_vision = 1.0  # citizen counts herself
        for c in self.neighbors:
            if (
                c.breed == "citizen"
                and c.condition == "Active"
                and c.jail_sentence == 0
            ):
                actives_in_vision += 1
        self.actives_in_vision = actives_in_vision
        self.arrest_probability = 1 - math.exp(
            -1
            * self.model.arrest_prob_constant
            * math.floor(cops_in_vision / actives_in_vision)
        )


class Cop(mesa.Agent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(self, unique_id, model, pos, vision):
        """
        Create a new Cop.
        Args:
            unique_id: unique int
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(unique_id, model)
        self.breed = "cop"
        self.pos = pos
        self.vision = vision

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if (
                agent.breed == "citizen"
                and agent.condition == "Active"
                and agent.jail_sentence == 0
            ):
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            sentence = self.random.randint(0, self.model.max_jail_term)
            arrestee.jail_sentence = sentence
            arrestee.condition = "Quiescent"
        if self.model.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)

    def update_neighbors(self):
        """
        Look around and see who my neighbors are.
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]
