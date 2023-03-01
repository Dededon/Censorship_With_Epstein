import mesa

from .model import EpsteinCivilViolence
from .agent import Citizen, Cop
from mesa.visualization.UserParam import Slider, Choice
from mesa.visualization.modules import ChartModule, TextElement


COP_COLOR = "#000000"
AGENT_QUIET_COLOR = "#648FFF"
AGENT_REBEL_COLOR = "#FE6100"
JAIL_COLOR = "#808080"
JAIL_SHAPE = "rect"


class JailChart(TextElement):
    """Display the current jail population."""

    def render(self, model):
        return f"Jail Population: {model.jail_count}"


class ActiveChart(TextElement):
    """Display the current active population."""

    def render(self, model):
        return f"Active Population: {model.active_count}"


class CitizenChart(TextElement):
    """Display the current active population."""

    def render(self, model):
        return f"Citizen Population: {model.citizen_count}"


class CopChart(TextElement):
    """Display the current active population."""

    def render(self, model):
        return f"Cop Population: {model.cop_count}"


class QuiescentChart(TextElement):
    """Display the current active population."""

    def render(self, model):
        return f"Quiescent Population: {model.quiescent_count}"


class AverageJailTerm(TextElement):
    """Display the average jail time."""
    
    def render(self, model):
        return f"Average Jail Term: {model.average_jail_term}"

class AverageStrongReceival(TextElement):
    """Display the strong tie receival"""

    def render(self, model):
        return f"Average Incoming Strong Ties Information: {model.strong_ties_receival}" #

class AverageWeakReceival(TextElement):
    """Display the strong tie receival"""

    def render(self, model):
        return f"Average Incoming Weak Ties Information: {model.weak_ties_receival}" # 

class GetActiveLink(TextElement):
    """Display the current active links"""

    def render(self, model):
        return f"Activated Links Ratio: {model.activated_link_ratio}" #

class GetCensorLink(TextElement):
    """Display the link censored at current timestep"""

    def render(self, model):
        return f"Censored Links Per step: {model.censored_links}" #

citizen_chart = CitizenChart()
cop_chart = CopChart()
quiescent_chart = QuiescentChart()
active_chart = ActiveChart()
jail_chart = JailChart()
average_jail_term = AverageJailTerm()
strong_receival = AverageStrongReceival()
weak_receival = AverageWeakReceival()
active_links = GetActiveLink()
censored_links = GetCensorLink()

chart = ChartModule(
    [
        {"Label": "Quiescent", "Color": "#648FFF"},
        {"Label": "Active", "Color": "#FE6100"},
        {"Label": "Jailed", "Color": "#808080"},
    ],
    data_collector_name="datacollector",
)


def citizen_cop_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "Shape": "circle",
        "x": agent.pos[0],
        "y": agent.pos[1],
        "Filled": "true",
    }

    if type(agent) is Citizen:
        color = (
            AGENT_QUIET_COLOR if agent.condition == "Quiescent" else AGENT_REBEL_COLOR
        )
        color = JAIL_COLOR if agent.jail_sentence else color
        shape = JAIL_SHAPE if agent.jail_sentence else "circle"
        portrayal["Color"] = color
        portrayal["Shape"] = shape
        if shape == "rect":
            portrayal["w"] = 0.9
            portrayal["h"] = 0.9
        else:
            portrayal["r"] = 0.5
            portrayal["Filled"] = "false"
        portrayal["Layer"] = 0

    elif type(agent) is Cop:
        portrayal["Color"] = COP_COLOR
        portrayal["r"] = 0.9
        portrayal["Layer"] = 1

    return portrayal

#activation_types = ["default","linear","quadratic","logistic"]
#activation_types = []
activation_types = ["With SWN", "Without SWN"]
# default model params do not match netlogo defaults and model does not exhibit
# the same behavior as netlogo
# model_params = dict(
#     height=40,
#     width=40,
#     citizen_density=0.7,
#     cop_density=0.074,
#     citizen_vision=7,
#     cop_vision=7,
#     legitimacy=0.8,
#     max_jail_term=1000,
# )

# matching netlogo defaults
model_params = dict(
    height=40,  #40
    width=40,   #40
    activation_type=Choice("With SWN or not?", "With SWN", activation_types),
    citizen_density=Slider("Initial Agent Density", 0.7, 0.0, 0.9, 0.1),
    cop_density=Slider("Initial Cop Density", 0.03, 0.0, 0.1, 0.01),
  #  citizen_vision=Slider("Citizen Vision", 5, 1, 10, 1),
  #  cop_vision=Slider("Cop Vision", 5, 1, 10, 1),
    legitimacy=Slider("Government Legitimacy", 0.75, 0.0, 1, 0.01),
    lattice_neighbor=Slider("Ring Lattice Neighbors", 4, 0, 5, 1),
    rewire_prob = Slider("Rewiring Rate", 0.5, 0, 1, 0.1),
    strong_weak_ratio=Slider("Strong-Weak Ratio", 5, 0, 10, 1),
    t_censor=Slider("Censor Timestep", 5, 0, 10, 1),
    p_link_censor=Slider("Link Censor Probability",0.0, 0, 0.4, 0.05),
)
canvas_element = mesa.visualization.CanvasGrid(citizen_cop_portrayal, 40, 40, 480, 480) # 40, 40, 480, 480
server = mesa.visualization.ModularServer(
    EpsteinCivilViolence,
    [
        canvas_element,
        citizen_chart,
        cop_chart,
        quiescent_chart,
        active_chart,
        jail_chart,
        average_jail_term,
        strong_receival,
        weak_receival,
        active_links,
        censored_links,
        chart,
    ],
    "Epstein Civil Violence",
    model_params,
)
