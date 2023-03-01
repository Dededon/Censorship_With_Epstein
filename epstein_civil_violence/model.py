import sched
import mesa
import random
import networkx as nx
from networkx.generators.random_graphs import watts_strogatz_graph
from .agent import Cop, Citizen


class EpsteinCivilViolence(mesa.Model):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full
    Attributes:
        height: grid height
        width: grid width
        citizen_density: approximate % of cells occupied by citizens.
        cop_density: approximate % of cells occupied by cops.
        citizen_vision: number of cells in each direction (N, S, E and W) that
            citizen can inspect
        cop_vision: number of cells in each direction (N, S, E and W) that cop
            can inspect
        legitimacy:  (L) citizens' perception of regime legitimacy, equal
            across all citizens
        max_jail_term: (J_max)
        active_threshold: if (grievance - (risk_aversion * arrest_probability))
            > threshold, citizen rebels
        arrest_prob_constant: set to ensure agents make plausible arrest
            probability estimates
        movement: binary, whether agents try to move at step end
        max_iters: model may not have a natural stopping point, so we set a
            max.
    """
    seed = random.random()

    schedule_types = {
        "Sequential": mesa.time.BaseScheduler,
        "Random": mesa.time.RandomActivation,
        "Simultaneous": mesa.time.SimultaneousActivation,
    }
    
    activation_types = ["With SWN", "Without SWN"]

    def __init__(
        self,
        width=40,
        height=40,
        citizen_density=0.7,
        cop_density=0.074,
        citizen_vision=7,
        cop_vision=7,
        legitimacy=0.8,
        max_jail_term=30,
        active_threshold=0.1,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=1000,
        schedule_type="Random",
        activation_type="With SWN",
        min_proportion=0.8,
        # SWN
        lattice_neighbor = 4,
        rewire_prob = 0.5,
        strong_weak_ratio = 5,
        t_censor = 5,
        p_link_censor = 0.10,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.citizen_density = citizen_density
        self.cop_density = cop_density

        self.citizen_vision = citizen_vision
        self.cop_vision = cop_vision

      #  self.cop_vision = citizen_vision    
        self.legitimacy = legitimacy
        self.max_jail_term = max_jail_term
        self.active_threshold = active_threshold
        self.arrest_prob_constant = arrest_prob_constant
        self.movement = movement
        self.max_iters = max_iters
        self.iteration = 0
        self.schedule = mesa.time.RandomActivation(self)
        self.schedule = self.schedule_types[schedule_type](self)
        self.activation_type = activation_type
        self.min_proportion = min_proportion
        self.grid = mesa.space.Grid(width, height, torus=True)

        # agent counts
        self.citizen_count = 0
        self.cop_count = 0
        self.jail_count = 0
        self.active_count = 0
        self.quiescent_count = 0
        self.average_jail_term = 0

        # SWN components
        self.strong_weak_ratio = strong_weak_ratio
        self.t_censor = t_censor
        self.p_link_censor = p_link_censor
        self.strong_ties_receival = 0
        self.weak_ties_receival = 0
        self.agent_dict = {}
        self.link_to_censor = set()
        self.active_links = 0
        self.activated_link_ratio = 0
        self.censored_links = 0

        model_reporters = {
            "Quiescent": lambda m: self.count_type_citizens(m, "Quiescent"),
            "Active": lambda m: self.count_type_citizens(m, "Active"),
            "Jailed": self.count_jailed,
            "Citizens": self.count_citizens,
            "Cops": self.count_cops,
            "Average Jailing Term": self.get_average_jail_term,
            "Average Strong Receival": self.get_average_strong_ties_receival,
            "Average Weak Receival": self.get_average_weak_ties_receival,
            "Active Links": self.get_active_link,
            "Activated Links Ratio": self.get_active_link_ratio,
            "Censored Links per step": self.get_censored_link,
        }
        agent_reporters = {
            "x": lambda a: a.pos[0],
            "y": lambda a: a.pos[1],
            "breed": lambda a: a.breed,
            "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
            "condition": lambda a: getattr(a, "condition", None),
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )
        unique_id = 0
        list_citizen_ids = []
        if self.cop_density + self.citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")
        for (contents, x, y) in self.grid.coord_iter():
            if self.random.random() < self.cop_density:
                cop = Cop(unique_id, self, (x, y), vision=self.cop_vision)
                unique_id += 1
                self.grid[x][y] = cop
                self.schedule.add(cop)
            elif self.random.random() < (self.cop_density + self.citizen_density):
                citizen = Citizen(
                    unique_id,
                    self,
                    (x, y),
                    hardship=self.random.random(),
                    regime_legitimacy=self.legitimacy,
                    risk_aversion=self.random.random(),
                    threshold=self.active_threshold,
                    vision=self.citizen_vision,
                )
                list_citizen_ids.append(unique_id)
                
                self.grid[x][y] = citizen
                self.schedule.add(citizen)
                self.agent_dict[unique_id] = citizen
                unique_id += 1


        # Initialize graph
        self.G = watts_strogatz_graph(len(list_citizen_ids), lattice_neighbor, rewire_prob)
        self.G = nx.relabel_nodes(self.G, dict(zip(self.G, list_citizen_ids)))
        nx.set_edge_attributes(self.G, 0, "censor_steps")
        self.number_of_edges = self.G.number_of_edges()

        self.running = True
        self.datacollector.collect(self)
        self.citizen_count = sum(value for value in self.count_agents(self).values())
        self.cop_count = self.count_cops(self)

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        # update the censorship terms
        self.censored_links = 0
        for u, v, data in list(self.G.edges.data()):
            if data['censor_steps'] > 0:
                self.G[u][v]['censor_steps'] -= 1
 
        self.schedule.step()
        # Censor links
      #  print(self.p_link_censor)
        self.active_links = len(self.link_to_censor)
        self.activated_link_ratio = self.get_active_link_ratio(self)

        for u, v in self.link_to_censor:
            if self.G[u][v]['censor_steps'] == 0:
                if self.random.random() < self.p_link_censor: #self.p_link_censo
                    self.G[u][v]['censor_steps'] = self.t_censor
                    self.censored_links += 1

        self.link_to_censor = set()
      #  print(self.censored_links)
        # collect data
        self.datacollector.collect(self)
        # update agent counts
        self.active_count = self.count_type_citizens(self, "Active")
        self.quiescent_count = self.count_type_citizens(self, "Quiescent")
        self.jail_count = self.count_jailed(self)
        self.average_jail_term = self.get_average_jail_term(self)
        self.strong_ties_receival  = self.get_average_strong_ties_receival(self)
        self.weak_ties_receival = self.get_average_weak_ties_receival(self)

        # update iteration 
        
        self.iteration += 1
        if self.iteration > self.max_iters:
            self.running = False

    @staticmethod
    def count_type_citizens(model, condition, exclude_jailed=True):
        """
        Helper method to count agents by Quiescent/Active.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "cop":
                continue
            if exclude_jailed and agent.jail_sentence > 0:
                continue
            if agent.condition == condition:
                count += 1
        return count

    @staticmethod
    def count_jailed(model):
        """
        Helper method to count jailed agents.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "citizen" and agent.jail_sentence > 0:
                count += 1
        return count

    @staticmethod
    def count_citizens(model):
        """
        Helper method to count citizens.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "citizen":
                count += 1
        return count

    @staticmethod
    def count_cops(model):
        """
        Helper method to count cops.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.breed == "cop":
                count += 1
        return count

    # combine all agent counts into one method
    @staticmethod
    def count_agents(model):
        """
        combines the various count methods into one
        """
        return {
            "Quiescent": model.count_type_citizens(model, "Quiescent"),
            "Active": model.count_type_citizens(model, "Active"),
            "Jailed": model.count_jailed(model),
        }

    @staticmethod
    def count_quiescent(model):
        return model.count_type_citizens(model, "Quiescent")

    @staticmethod
    def count_active(model):
        return model.count_type_citizens(model, "Active")

    @staticmethod
    def get_active_link(model):
        return model.active_links

    @staticmethod
    def get_active_link_ratio(model):
        return model.active_links / model.number_of_edges

    @staticmethod
    def get_censored_link(model):
        return model.censored_links

    @staticmethod
    def get_average_jail_term(model):
        """
        Helper method to calculate average jail term.
        """
        total = 0
        for agent in model.schedule.agents:
            if agent.breed == "citizen" and agent.jail_sentence > 0:
                total += agent.jail_sentence
        if model.jail_count == 0:
            return 0
        else:
            return total / model.jail_count

    @staticmethod
    def get_average_strong_ties_receival(model):
       """
       Helper method to calculate strong tie receival per step
       """
       total = 0
       count = 0
       for agent in model.schedule.agents:
           if agent.breed == "citizen" and agent.jail_sentence == 0:
               total += agent.strong_ties_received
               count += 1
       if total == 0:
           return 0
       else:
           return total / count

    @staticmethod
    def get_average_weak_ties_receival(model):
       """
       Helper method to calculate strong tie receival per step
       """
       total = 0
       count = 0
       for agent in model.schedule.agents:
           if agent.breed == "citizen" and agent.jail_sentence == 0:
               total += agent.weak_ties_received
               count += 1
       if total == 0:
           return 0
       else:
           return total / count

