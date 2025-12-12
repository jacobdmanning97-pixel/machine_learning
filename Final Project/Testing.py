import requests
import time
import pandas as pd
import numpy as np
import pickle
import random
from typing import Dict
from bs4 import BeautifulSoup
import re

def get_pokemon_info(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    #url = f"https://pokeapi.co/api/v2/move/{move_name.lower()}/""
    response = requests.get(url)
    bst = 0

    if response.status_code == 200:
        pokemon_data = response.json()
        base_stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        base_stat_numbers = [stat["base_stat"] for stat in pokemon_data["stats"]]
        pokemon_info = {
            "name": pokemon_data["name"],
            "base_stats": {base_stat_names[i]: base_stat_numbers[i] for i in range(len(base_stat_names))},
            "abilities": [ability["ability"]["name"] for ability in pokemon_data["abilities"]],
            "types": [type_data["type"]["name"] for type_data in pokemon_data["types"]],
            "moves": [move["move"]["name"] for move in pokemon_data["moves"]],
        }
        pokemon_info["def_coverage"] = type_chart.def_coverage(pokemon_info["types"])
        pokemon_info["stab_coverage"] = type_chart.get_stab_coverage(pokemon_info["types"])

        for stat, value in pokemon_info["base_stats"].items():
            bst += value
        pokemon_info["base_stat_total"] = bst

        return pokemon_info
    else:
        return None
    
def display_pokemon_info(pokemon_info, coverage):
    if pokemon_info:
        print(f"Name: {pokemon_info['name']}")
        print(f"Stats: {', '.join(map(str,pokemon_info['stats']))}")
        print(f"Abilitie(s): {', '.join(pokemon_info['abilities'])}")
        print(f"Type(s): {', '.join(pokemon_info['types'])}")
        for i in coverage:
            print(i,":",", ".join(coverage[i]))
        #print(f"Moves: {', '.join(pokemon_info['moves'])}")
    else:
        print("Pokémon not found!")
        print()

def get_move_info(move_name):
    if move_name not in move_dex.move_database:
        url = f"https://pokeapi.co/api/v2/move/{move_name.lower()}/"
        response = requests.get(url)
        time.sleep(0.1)
        if response.status_code == 200:
            move_data = response.json()
            if move_data["meta"] == None:
                    move_info = {
                        "name": move_data["name"],
                    }
            else:
                move_info = {
                    "name": move_data["name"],
                    "accuracy": move_data["accuracy"],
                    "damage_type": move_data["damage_class"]["name"],
                    "power": move_data["power"],
                    "priority": move_data["priority"],
                    "target": move_data["target"]["name"],
                    "type": move_data["type"]["name"],
                    "ailment_name": move_data["meta"]["ailment"]["name"],
                    "ailment_chance": move_data["meta"]["ailment_chance"],
                    "crit_rate": move_data["meta"]["crit_rate"],
                    "drain": move_data["meta"]["drain"],
                    "flinch_chance": move_data["meta"]["flinch_chance"],
                    "healing": move_data["meta"]["healing"],
                    "max_hits": move_data["meta"]["max_hits"],
                    "max_turns": move_data["meta"]["max_turns"],
                    "min_hits": move_data["meta"]["min_hits"],
                    "min_turns": move_data["meta"]["min_turns"],
                    "stat_chance": move_data["meta"]["stat_chance"],
                    "stat_changes": move_data["stat_changes"],
                }
            return move_info
        else:
            return None

class Type_Chart:
    def __init__(self):
        self.types = [
            'normal', 'fire', 'water', 'electric', 'grass', 'ice',
            'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
            'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
        ]
        
        # Initialize effectiveness matrix 
        self.effectiveness = np.ones((18, 18))
        
        self.setup_type_chart()
    
    def setup_type_chart(self):
        super_effective = {
            'normal': [],
            'fire': ['grass', 'ice', 'bug', 'steel'],
            'water': ['fire', 'ground', 'rock'],
            'electric': ['water', 'flying'],
            'grass': ['water', 'ground', 'rock'],
            'ice': ['grass', 'ground', 'flying', 'dragon'],
            'fighting': ['normal', 'ice', 'rock', 'dark', 'steel'],
            'poison': ['grass', 'fairy'],
            'ground': ['fire', 'electric', 'poison', 'rock', 'steel'],
            'flying': ['grass', 'fighting', 'bug'],
            'psychic': ['fighting', 'poison'],
            'bug': ['grass', 'psychic', 'dark'],
            'rock': ['fire', 'ice', 'flying', 'bug'],
            'ghost': ['psychic', 'ghost'],
            'dragon': ['dragon'],
            'dark': ['psychic', 'ghost'],
            'steel': ['ice', 'rock', 'fairy'],
            'fairy': ['fighting', 'dragon', 'dark']
        }

        not_very_effective = {
            'normal': ['rock', 'steel'],
            'fire': ['fire', 'water', 'rock', 'dragon'],
            'water': ['water', 'grass', 'dragon'],
            'electric': ['electric', 'grass', 'dragon'],
            'grass': ['fire', 'grass', 'poison', 'flying', 'bug', 'dragon', 'steel'],
            'ice': ['fire', 'water', 'ice', 'steel'],
            'fighting': ['poison', 'flying', 'psychic', 'bug', 'fairy'],
            'poison': ['poison', 'ground', 'rock', 'ghost'],
            'ground': ['grass', 'bug'],
            'flying': ['electric', 'rock', 'steel'],
            'psychic': ['psychic', 'steel'],
            'bug': ['fire', 'fighting', 'poison', 'flying', 'ghost', 'steel', 'fairy'],
            'rock': ['fighting', 'ground', 'steel'],
            'ghost': ['dark'],
            'dragon': ['steel'],
            'dark': ['fighting', 'dark', 'fairy'],
            'steel': ['fire', 'water', 'electric', 'steel'],
            'fairy': ['fire', 'poison', 'steel']
        }
        
        immune = {
            'normal': ['ghost'],
            'fire': [],
            'water': [],
            'electric': ['ground'],
            'grass': [],
            'ice': [],
            'fighting': ['ghost'],
            'poison': ['steel'],
            'ground': ['flying'],
            'flying': [],
            'psychic': ['dark'],
            'bug': [],
            'rock': [],
            'ghost': ['normal'],
            'dragon': ['fairy'],
            'dark': [],
            'steel': [],
            'fairy': []
        }
        
        # Apply the effectiveness rules to the matrix
        for i, attack_type in enumerate(self.types):
            # Apply super effective
            for defense_type in super_effective[attack_type]:
                j = self.types.index(defense_type)
                self.effectiveness[i, j] = 2.0
            
            # Apply not very effective
            for defense_type in not_very_effective[attack_type]:
                j = self.types.index(defense_type)
                self.effectiveness[i, j] = 0.5
            
            # Apply immune
            for defense_type in immune[attack_type]:
                j = self.types.index(defense_type)
                self.effectiveness[i, j] = 0.0
   
    def effectiveness_internal(self, attack_type, defense_type):
        i = self.types.index(attack_type)
        j = self.types.index(defense_type)
        return self.effectiveness[i, j]
    
    def effectiveness_calc_def(self, attack_type, defense_type1, defense_type2=None):
        if defense_type2 is None:
            return self.effectiveness_internal(attack_type, defense_type1)
        
        eff1 = self.effectiveness_internal(attack_type, defense_type1)
        eff2 = self.effectiveness_internal(attack_type, defense_type2)
        return eff1 * eff2
    
    def effectiveness_calc_atk(self, attack_type1, attack_type2, defense_type):
        if attack_type2 is None:
            return self.effectiveness_internal(attack_type1, defense_type)
        
        eff1 = self.effectiveness_internal(attack_type1, defense_type)
        eff2 = self.effectiveness_internal(attack_type2, defense_type)
        return [eff1, eff2]
 
    def get_stab_coverage(self, stab_type):
        immune_types = []
        resistant_types = []
        strong_types = []


        stab1 = stab_type[0]
        if len(stab_type) > 1:
            stab2 = stab_type[1]
        else:
            stab2 = None

        if stab2 == None:
            for j, defense_type in enumerate(self.types):
                effect = self.effectiveness_calc_atk(stab1, stab2, defense_type)
                if effect == 0:
                    immune_types.append(defense_type)
                if effect == 0.5:
                    resistant_types.append(defense_type)
                if effect == 2.0:
                    strong_types.append(defense_type)
        else:
            for j, defense_type in enumerate(self.types):
                [effect1, effect2] = self.effectiveness_calc_atk(stab1, stab2, defense_type)
                if effect1 == 0 and effect2 == 0:
                    immune_types.append(defense_type)
                if effect1 == 0.5 and effect2 == 0.5:
                    resistant_types.append(defense_type)
                elif effect1 == 0.5 and effect2 == 0:
                    resistant_types.append(defense_type)
                elif effect1 == 0 and effect2 == 0.5:
                    resistant_types.append(defense_type)
                if effect1 == 2.0 or effect2 == 2.0:
                    strong_types.append(defense_type)

        coverage = {
            "Immune": [],
            "Resistant": [],
            "Strong": [],
        }

        coverage["Immune"] = immune_types
        coverage["Resistant"] = resistant_types
        coverage["Strong"] = strong_types

        return coverage
    
    def def_coverage(self, defense_type):
        immune_types = []
        super_resistant_types = []
        resistant_types = []
        weak_types = []
        super_weak_types = []

        defense_type1 = defense_type[0]
        if len(defense_type) > 1:
            defense_type2 = defense_type[1]
        else:
            defense_type2 = None

        for attack_type in self.types:
            effectiveness_value = self.effectiveness_calc_def(attack_type, defense_type1, defense_type2)
            if effectiveness_value == 0:
                immune_types.append(attack_type)
            if effectiveness_value == 0.5:
                resistant_types.append(attack_type)
            if effectiveness_value == 0.25:
                super_resistant_types.append(attack_type)
            if effectiveness_value == 2.0:
                weak_types.append(attack_type)
            if effectiveness_value == 4.0:
                super_weak_types.append(attack_type)

        coverage = {
            "Immune": [],
            "Resistant": [],
            "Super Resistant": [],
            "Weak": [],
            "Super Weak": [],
        }

        coverage["Immune"] = immune_types
        coverage["Resistant"] = resistant_types
        coverage["Super Resistant"] = super_resistant_types
        coverage["Weak"] = weak_types
        coverage["Super Weak"] = super_weak_types

        return coverage

class Pokedex:
    def __init__(self):
        self.pokemon_database = {}
    
    def register_pokemon(self, poke_info):
        """Register a Pokemon in the database"""
        if poke_info['name'].lower() not in self.pokemon_database:
            self.pokemon_database[poke_info['name'].lower()] = poke_info
        else:
            print("Pokemon already in database")
    
    def load_data(self, database):
        self.pokemon_database = database

    def load_instance(self, loaded):
        pokemon_name = loaded['name'].lower()
        if pokemon_name not in self.pokemon_database:
            return f"Pokemon {pokemon_name} not found in database"
        else:
            base_data = self.pokemon_database[pokemon_name]
            base_stats = base_data['base_stats']
            nature = loaded['nature']
            tera = loaded['tera_type']
            evs = loaded['evs']
            ivs = loaded['ivs']
            stats = self._calculate_final_stats(base_stats, evs, ivs, nature)
            bst = base_data['base_stat_total']
            roles = self.define_role(base_stats, bst)
            move_names = loaded['moves']

            if evs is None:
                evs = self._generate_random_evs(nature)

            return {
                'name': pokemon_name,
                'types': base_data['types'],
                'tera': tera,
                'roles': roles,
                'defensive_coverage': base_data['def_coverage'],
                'stab_coverage': base_data['stab_coverage'],
                'nature': nature,
                'evs': evs,
                'bst': bst,
                'max_stat': max(stats, key = stats.get),
                'min_stat': min(stats, key = stats.get),
                'stats': stats,
                'base_stats': base_stats,
                'moves': move_names
            }

    def create_random_instance(self, pokemon_name):
        """Create random instance of a registered Pokemon"""
        if pokemon_name not in self.pokemon_database:
            raise ValueError(f"Pokemon {pokemon_name} not found in database")
        
        base_data = self.pokemon_database[pokemon_name]
        return self._create_pokemon_instance(base_data)

    def define_role(self, base_stats, bst):
        roles = []

        max_atk = max(['attack', 'sp_attack'], key = lambda stat: base_stats[stat])
        bulk_score = (base_stats['hp'] + base_stats['defense'] + base_stats['sp_defense'])/3

        if bst/6 <= base_stats[max_atk]:
            roles.append(f'{max_atk}er')
        if bst/6 <= bulk_score:
            roles.append('bulky')
        if bst/6 <= base_stats['speed']:
            roles.append('speedy')

        return roles

    def _create_pokemon_instance(self, pokemon_data):
        """Create a Pokemon instance with random EVs"""
        types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
        base_stats = pokemon_data['base_stats']
        # Random nature
        nature = self._get_random_nature()   
        bst = pokemon_data['base_stat_total']
        roles = self.define_role(base_stats, bst)
        # Generate random EVs
        evs = self._generate_random_evs(nature)
        if 'sp_attacker' in roles:
            ivs = {'hp': 31, 'attack': 0, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}
        else:
            ivs = {'hp': 31, 'attack': 31, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}
        stats = self._calculate_final_stats(base_stats, evs, ivs, nature)

        moves = move_dex.choose_move_set(roles, pokemon_data["moves"], random.choice(['True','False']))
        move_names = [move['name'] for move in moves]

        return {
            'name': pokemon_data['name'],
            'types': pokemon_data['types'],
            'tera': random.choice(types),
            'roles': roles,
            'defensive_coverage': pokemon_data['def_coverage'],
            'stab_coverage': pokemon_data['stab_coverage'],
            'nature': nature,
            'evs': evs,
            'bst': bst,
            'max_stat': max(stats, key = stats.get),
            'min_stat': min(stats, key = stats.get),
            'stats': stats,
            'base_stats': base_stats,
            'moves': move_names
        }
    
    def _generate_random_evs(self, nature) -> Dict[str, int]:
        """Generate random EV spread that sums to 510 or less"""
        stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        evs = {stat: 0 for stat in stats}
        nature_stats = list(self._get_nature_modifiers(nature))
        if len(nature_stats) == 1:
            prioritized_stats = random.sample(stats, random.randint(2, 3))
        else: 
            stats.pop(stats.index(nature_stats[0]))
            stats.pop(stats.index(nature_stats[1]))
            prioritized_stats = random.sample(stats, random.randint(1, 2))
            prioritized_stats.append(nature_stats[0])
        leftover_stats = [stat for stat in stats if stat not in prioritized_stats]

        remaining_evs = 510
        # Distribute EVs to prioritized stats first (in multiples of 8 for efficiency)
        for stat in prioritized_stats:
            if remaining_evs <= 0:
                break
            
            # At level 50, we want EVs in multiples of 8 for efficiency
            max_possible = min(252, remaining_evs)
            # Round down to nearest multiple of 8
            max_evs = (max_possible // 8) * 8

            if len(prioritized_stats) == 3:
                if max_evs > 168:
                    # Choose a multiple of 8 between 0 and max_evs
                    ev_options = list(range(104, 169, 8))
                    chosen_evs = random.choice(ev_options)
                    evs[stat] = chosen_evs
                    remaining_evs -= chosen_evs
            else:
                if max_evs > 168:
                    # Choose a multiple of 8 between 0 and max_evs
                    ev_options = list(range(168, max_evs + 1, 8))
                    chosen_evs = random.choice(ev_options)
                    evs[stat] = chosen_evs
                    remaining_evs -= chosen_evs
        
        # Distribute remaining EVs randomly in efficient chunks
        while remaining_evs > 0:
            stat = random.choice(leftover_stats)
            # Add in chunks of 8 EVs for efficiency at level 50
            max_add = min(252 - evs[stat], remaining_evs)
            max_add = (max_add // 8) * 8  # Round down to multiple of 8
            
            if max_add > 0:
                add_evs = random.choice([8, 16, 24, 32, 40, 48])  # Common efficient amounts
                add_evs = min(add_evs, max_add)
                evs[stat] += add_evs
                remaining_evs -= add_evs
            else:
                # If we can't add efficient chunks, just use what's left
                evs[stat] += remaining_evs
                remaining_evs = 0
        return evs
    
    def _calculate_final_stats(self, base_stats: Dict[str, int], evs: Dict[str, int],  ivs: Dict[str, int],
                            nature: str) -> Dict[str, int]:
        """Calculate final stats considering base stats, EVs, IVs, nature, and level"""
        stats = {}
        nature_modifiers = self._get_nature_modifiers(nature)
        for stat, base in base_stats.items():
            if stat == 'hp':
                stats[stat] = self._calculate_hp_stat(base, evs[stat])
            else:
                stats[stat] = self._calculate_other_stat(base, evs[stat], ivs[stat], nature_modifiers.get(stat, 1.0))
        return stats
    
    def _calculate_hp_stat(self, base: int, ev: int) -> int:
        """Calculate HP stat"""
        # HP formula: ((2 * Base + IV + EV/4) * Level) / 100 + Level + 10
        return ((2 * base + 31 + ev // 4) * 50) // 100 + 50 + 10
    
    def _calculate_other_stat(self, base: int, ev: int, iv: int, nature_modifier: float) -> int:
        """Calculate other stats (Attack, Defense, etc.)"""
        # Other stats formula: (((2 * Base + IV + EV/4) * 50) / 100 + 5) * Nature
        stat_value = ((2 * base + iv + ev // 4) * 50) // 100 + 5
        return int(stat_value * nature_modifier)
    
    def _get_random_nature(self) -> str:
        """Get a random nature"""
        natures = [
            'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
            'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
            'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
            'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
            'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
        ]
        return random.choice(natures)
    
    def _get_nature_modifiers(self, nature: str) -> Dict[str, float]:
        """Get stat modifiers for a given nature"""
        # Nature effects: +10% to one stat, -10% to another
        nature_effects = {
            'Hardy': {'attack': 1.0, 'attack': 1.0},
            'Lonely': {'attack': 1.1, 'defense': 0.9},
            'Brave': {'attack': 1.1, 'speed': 0.9},
            'Adamant': {'attack': 1.1, 'sp_attack': 0.9},
            'Naughty': {'attack': 1.1, 'sp_defense': 0.9},
            'Bold': {'defense': 1.1, 'attack': 0.9},
            'Docile': {'defense': 1.0, 'defense': 1.0},
            'Relaxed': {'defense': 1.1, 'speed': 0.9},
            'Impish': {'defense': 1.1, 'sp_attack': 0.9},
            'Lax': {'defense': 1.1, 'sp_defense': 0.9},
            'Timid': {'speed': 1.1, 'attack': 0.9},
            'Hasty': {'speed': 1.1, 'defense': 0.9},
            'Serious': {'speed': 1.0, 'speed': 1.0},
            'Jolly': {'speed': 1.1, 'sp_attack': 0.9},
            'Naive': {'speed': 1.1, 'sp_defense': 0.9},
            'Modest': {'sp_attack': 1.1, 'attack': 0.9},
            'Mild': {'sp_attack': 1.1, 'defense': 0.9},
            'Quiet': {'sp_attack': 1.1, 'speed': 0.9},
            'Bashful': {'sp_attack': 1.0, 'sp_attack': 1.0},
            'Rash': {'sp_attack': 1.1, 'sp_defense': 0.9},
            'Calm': {'sp_defense': 1.1, 'attack': 0.9},
            'Gentle': {'sp_defense': 1.1, 'defense': 0.9},
            'Sassy': {'sp_defense': 1.1, 'speed': 0.9},
            'Careful': {'sp_defense': 1.1, 'sp_attack': 0.9},
            'Quirky' : {'sp_defense': 1.0, 'sp_defense': 1.0}
        }
        return nature_effects.get(nature, {})
    
    def create_multiple_instances(self, pokemon_name: str, count: int = 1):
        """Create multiple random instances of a Pokemon"""
        return [self.create_random_instance(pokemon_name.lower()) for _ in range(count)]

    def print_pokemon_details(self, pokemon_instance: Dict):
        """Pretty print Pokemon details"""
        print(f"\n=== {pokemon_instance['name'].capitalize()} ===")
        print(f"Type: {', '.join(pokemon_instance['types'])}")
        print("\nDefensive Coverage:")
        for type, coverage in pokemon_instance['defensive_coverage'].items():
            print(f"  {type.upper()}: {", ".join(coverage)}")
        print("\nStab Coverage:")
        for type, coverage in pokemon_instance['stab_coverage'].items():
            print(f"  {type.upper()}: {", ".join(coverage)}")
        print("\nRoles")
        print(f"  {", ".join(pokemon_instance['roles'])}")
        print(f"\nNature: {pokemon_instance['nature']}")
        print("EVs:")
        ev_tot = 0
        for stat, ev in pokemon_instance['evs'].items():
            print(f"  {stat.upper()}: {ev}")
            ev_tot += ev
        print("Total EVs: ",ev_tot)
        print("\nStats:")
        for stat, value in pokemon_instance['stats'].items():
            print(f"  {stat.upper()}: {value}")
        print(f"BST: {pokemon_instance['bst']}, Max Stat: {pokemon_instance['max_stat']}, Min Stat: {pokemon_instance['min_stat']}")
        print("\nMoves:")
        for move in pokemon_instance['moves']:
            print(f"  {move.replace('-',' ').capitalize()}")

class Move_Dex:
    def __init__(self):
        self.move_database = {}
    
    def register_move(self, move_info):
        """Register a Move in the database"""
        self.move_database[move_info['name'].lower()] = move_info

    def load_moves(self, data):
        self.move_database = data

    def choose_move_set(self, roles, move_list = None, status_flag = True):
        status_moves = []
        physical_moves = []
        special_moves = []
        moves = []

        for move in self.move_database:
            if move in move_list:
                if self.move_database[move]['damage_type'] == 'status':
                    status_moves.append(self.move_database[move])
                if self.move_database[move]['damage_type'] == 'physical':
                    physical_moves.append(self.move_database[move])
                if self.move_database[move]['damage_type'] == 'special':
                    special_moves.append(self.move_database[move])

        if status_flag == False:
            atk = 4
        elif 'bulky' in roles:
            atk = random.randint(1,3)
        else:
            atk = random.randint(2,4)

        if 'attacker' in roles:
            rest = 4 - atk
            moves = random.sample(physical_moves, k = atk)
            if rest > 0:
                moves.extend(random.sample(status_moves, k = rest))
        elif 'sp_attacker' in roles:
            rest = 4 - atk
            moves = random.sample(special_moves, k = atk)
            if rest > 0:
                moves.extend(random.sample(status_moves, k = rest))
        else:
            rest = 4 - atk
            moves = random.sample(status_moves, k = atk)
            if rest > 0:
                moves.extend(random.sample((special_moves + physical_moves), k = rest))
        
        return moves

    def display_move_info(self, move_name):
        move_info = self.move_database[move_name]
        print(f"Name: {move_info['name']}")
        print(f"Accuracy: {move_info['accuracy']}")
        print(f"Damage Type: {move_info['damage_type']}")
        print(f"Power: {move_info['power']}")
        print(f"Priority: {move_info['priority']}")
        print(f"Target: {move_info['target']}")
        print(f"Type: {move_info['type']}")
        print(f"Ailment: {move_info['ailment_name']}")
        print(f"Ailment Chance: {move_info['ailment_chance']}")
        print(f"Crit Rate: {move_info['crit_rate']}")
        print(f"Drain: {move_info['drain']}")
        print(f"Flinch Chance: {move_info['flinch_chance']}")
        print(f"Healing: {move_info['healing']}")
        print(f"Max Hits: {move_info['max_hits']}")
        print(f"Max Turns: {move_info['max_turns']}")
        print(f"Min Hits: {move_info['min_hits']}")
        print(f"Min Turns: {move_info['min_turns']}")
        print(f"Stat Chance: {move_info['stat_chance']}")
        if len(move_info['stat_changes']) > 0:
            for i in range(len(move_info['stat_changes'])):
                dict = move_info['stat_changes'][i]
                print(f"Stat Changes: {dict['change']} {dict['stat']['name']}")
        else:
            print("Stat Changes: None")

    def rate_move(self, move_name, poke_info):
        move_info = self.get_move_info(move_name)
        rate = 1

        if move_info['damage_type'] != 'status':
            if move_info['type'] in poke_info['types']:
                rate *= 1.1

        if move_info['accuracy'] != None:
            if float(move_info['accuracy']) > 0:
                rate *= float(move_info['accuracy'])/100

        if move_info['power'] != None:
            if float(move_info['power']) > 0:
                rate *= (1+float(move_info['power'])/100)

        if float(move_info['priority']) != 0:
            if float(move_info['priority']) > 0:
                rate *= (1+float(move_info['priority'])/10)
            if float(move_info['priority']) < 0:
                rate *= (1-float(move_info['priority'])/10)

        if move_info['target'] != 'selected-pokemon':
            rate *= 1.5

        if move_info['ailment_name'] != 'none':
            rate *= (1+float(move_info['ailment_chance'])/100)
        
        if move_info['crit_rate'] != 0:
            rate *= (1+float(move_info['crit_rate']))

        if move_info['drain'] != 0:
            rate *= (1+float(move_info['drain'])/100)

        if move_info['flinch_chance'] != 0:
            rate *= (1+float(move_info['flinch_chance'])/100)

        if move_info['healing'] != 0:
            rate *= (1+float(move_info['healing'])/100)

        if move_info['max_hits'] != None:
            if float(move_info['max_hits']) > 0:
                rate *= float(move_info['max_hits'])*(float(move_info['accuracy'])/100)**(float(move_info['max_hits'])-1)

        if len(move_info['stat_changes']) != 0:
            for stat in move_info['stat_changes']:
                if float(stat['change']) < 0:
                    rate *= abs(0.9*float(stat['change']))
                else:
                    rate *= 1.1*float(stat['change'])

        if 'attacker' in poke_info['roles']:
            if move_info['damage_type'] == 'physical':
                rate *= 1.5

        if 'sp_attacker' in poke_info['roles']:
            if move_info['damage_type'] == 'special':
                rate *= 1.5

        if 'bulky' in poke_info['roles']:
            if move_info['damage_type'] == 'status':
                rate *= 1.5

        return rate

    def get_move_info(self, move_name):
        return self.move_database[move_name]

class Teams:
    def __init__(self):
        self.team = {}
        self.member_names = []

    def create_team(self, name_list):
        self.member_names = random.sample(name_list, k = 6)
        i=0
        for member in self.member_names:
            self.team[f'Member {i+1}'] = pokedex.create_multiple_instances(member, count = 1)[0]
            i += 1

    def load_team(self, team_list):
        #homogenize names
        #names = [indeedee, basculegon, maushold, ...]
        for poke in range(6):
            self.team[f'Member {poke + 1}'] = pokedex.load_instance(team_list[f'Member {poke + 1}'])
            self.member_names.append(team_list[f'Member {poke + 1}']['name'])

    def print_team_details(self):
        for index, member in self.team.items():
            pokedex.print_pokemon_details(member)

    def role_composition(self):
        roles = []
        for index, member in self.team.items():
            roles.extend(member['roles'])
        roles = {'attacker': roles.count('attacker'),
        'sp_attacker': roles.count('sp_attacker'),
        'bulky': roles.count('bulky'),
        'speedy': roles.count('speedy')
        }
        return roles

    def speed_control(self):
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        spd_moves = ['icy-wind', 'trick-room', 'tailwind', 'electroweb', 'bulldoze', 'thunder-wave', 'dragon-dance', 'quiver-dance']
        spd_count = []
        for poke in moves:
            for move in poke:
                if move in spd_moves:
                    spd_count.append(move)
        return len(spd_count)

    def pivoting_moves(self):
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        pivot_moves = ['baton-pass', 'chilly-reception', 'flip-turn', 'parting-shot', 'shed-tail', 'teleport', 'u-turn', 'volt-switch']
        pivot_count = []
        for poke in moves:
            for move in poke:
                if move in pivot_moves:
                    pivot_count.append(move)
        return len(pivot_count)

    def move_scores(self):
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        move_scores = []
        i = 0
        for poke in moves:
            move_score = []
            for move in poke:
                move_score.append(move_dex.rate_move(move, self.team[f'Member {i+1}']))
            i += 1
            move_scores.append(float(np.mean(move_score)))
        return move_scores

    def weather(self):
        weather_setters = {
        'rain': ['Pelipper', 'Politoed'],
        'sun': ['Torkoal'],
        'hail': ['Ninetales-Alola', 'Abomasnow'],
        'sand': ['Tyranitar', 'Hippowdon']
        }
        weather_moves = {'sunny-day': 'sun', 'rain-dance': 'rain', 'snowscape': 'hail', 'chilly-reception': 'hail', 'sandstorm': 'sand'}
        recover = ['synthesis', 'morning-sun', 'moonlight']
        sun = ['solar-beam', 'solar-blade', 'growth']
        rain = ['thunder', 'hurricane']

        abusers = {
        'rain': ['Archaludon', 'Volcarona'],
        'sun': [],
        'hail': [],
        'sand': []
        }

        if 'Archaludon' not in self.member_names:
            abusers['rain'].remove('Archaludon')
        if 'Volcarona' not in self.member_names:
            abusers['rain'].remove('Volcarona')

        for index, member in self.team.items():
            types = member['types']
            for type in types:
                if type == 'water':
                    abusers['rain'].append(member['name'])
                if type == 'fire':
                    abusers['sun'].append(member['name'])
                if type == 'ice':
                    abusers['hail'].append(member['name'])
                if type == 'steel':
                    abusers['sand'].append(member['name'])
                if type == 'rock':
                    abusers['sand'].append(member['name'])
                if type == 'ground':
                    abusers['sand'].append(member['name'])

        weather_types = []
        for weather, setters in weather_setters.items():
            weather_types.extend([weather for setter in setters if setter in self.member_names])

        tot_moves = []
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        for move_list in moves:
            tot_moves.extend(move_list)

        for move in tot_moves:
            if move in weather_moves:
                weather_types.append(weather_moves[move])
            
        move_dict = {}
        for index, member in self.team.items():
            for move in member['moves']:
                move_dict[move] = move_dex.get_move_info(move)['type']

        rain_improved = []
        rain_unimproved = []
        sun_improved = []
        sun_unimproved = []
        hail_improved = []
        hail_unimproved = []
        sand_unimproved = []

        if 'rain' in weather_types:
            for move in move_dict:
                if move in sun:
                    rain_unimproved.append(move)
                if move in rain:
                    rain_improved.append(move)
                if move in recover:
                    rain_unimproved.append(move)
                if move_dict[move] == 'water':
                    rain_improved.append(move)
                if move_dict[move] == 'fire':
                    rain_unimproved.append(move)

        if 'sun' in weather_types:
            for move in move_dict:
                if move in sun:
                    sun_improved.append(move)
                if move in rain:
                    sun_unimproved.append(move)
                if move in recover:
                    sun_improved.append(move)
                if move_dict[move] == 'fire':
                    sun_improved.append(move)
                if move_dict[move] == 'water':
                    sun_unimproved.append(move)

        if 'hail' in weather_types:
            for move in move_dict:
                if move == 'blizzard':
                    hail_improved.append(move)
                if move == 'aurora-veil':
                    hail_improved.append(move)
                if move in recover:
                    hail_unimproved.append(move)

        if 'sand' in weather_types:
            for move in move_dict:
                if move == 'solar-beam':
                    sand_unimproved.append(move)
                if move in recover:
                    sand_unimproved.append(move)
        
        weather_abuse = 0
        for weather in set(weather_types):
            weather_abuse += len(abusers[weather])
            #print(abusers[weather])

        #print(rain_improved, rain_unimproved, sun_improved, sun_unimproved, hail_improved, hail_unimproved, sand_unimproved)

        weather_abuse += len(rain_improved)
        weather_abuse -= len(rain_unimproved)
        weather_abuse += len(sun_improved)
        weather_abuse -= len(sun_unimproved)
        weather_abuse += len(hail_improved)
        weather_abuse -= len(hail_unimproved)
        weather_abuse -= len(sand_unimproved)

        return weather_abuse
                
    def terrain(self):
        terrain_setters = {
        'grass': ['Rillaboom'],
        'psychic': ['Indeedee-male', 'Indeedee-female'],
        }
        terrain_moves = {'grassy-terrain': 'grass', 'psychic-terrain': 'psychic', 'electric-terrain': 'electric', 'misty-terrain': 'mist'}

        grass = ['nature-power', 'terrain-pulse', 'grassy-glide', 'earthquake', 'bulldoze', 'magnitude']
        psychic = ['nature-power', 'terrain-pulse', 'expanding-force']
        electric = ['nature-power', 'rising-voltage', 'terrain-pulse']
        mist = ['nature-power', 'terrain-pulse']

        terrain_types = []
        for terrain, setters in terrain_setters.items():
            terrain_types.extend([terrain for setter in setters if setter in self.member_names])

        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        tot_moves = []
        for move_list in moves:
            tot_moves.extend(move_list)
        for move in tot_moves:
            if move in terrain_moves:
                terrain_types.append(terrain_moves[move])

        move_dict = {}
        for index, member in self.team.items():
            for move in member['moves']:
                move_dict[move] = move_dex.get_move_info(move)['type']

        #print(terrain_types)

        grass_improved = []
        grass_unimproved = []
        psychic_improved = []
        electric_improved = []
        mist_improved = []
        mist_unimproved = []

        if 'grass' in terrain_types:
            for move in move_dict:
                if move in grass[0:3]:
                    grass_improved.append(move)
                if move in grass[3:6]:
                    grass_unimproved.append(move)
                if move_dict[move] == 'grass':
                    grass_improved.append(move)

        if 'psychic' in terrain_types:
            for move in move_dict:
                if move in psychic:
                    psychic_improved.append(move)
                if move_dict[move] == 'psychic':
                    psychic_improved.append(move)

        if 'electric' in terrain_types:
            for move in move_dict:
                if move in electric:
                    electric_improved.append(move)
                if move_dict[move] == 'electric':
                    electric_improved.append(move)

        if 'mist' in terrain_types:
            for move in move_dict:
                if move in mist:
                    mist_improved.append(move)
                if move_dict[move] == 'dragon':
                    mist_unimproved.append(move)

        #print(grass_improved, grass_unimproved, psychic_improved, electric_improved, mist_improved, mist_unimproved)

        terrain_score = 0

        terrain_score += len(grass_improved)
        terrain_score -= len(grass_unimproved)
        terrain_score += len(psychic_improved)
        terrain_score += len(electric_improved)
        terrain_score += len(mist_improved)
        terrain_score -= len(mist_unimproved)

        return terrain_score

    def screens(self):
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        screen_moves = ['reflect', 'light-screen', 'aurora-veil']
        screen_count = []
        for poke in moves:
            for move in poke:
                if move in screen_moves:
                    screen_count.append(move)
        return len(screen_count)

    def random(self):
        score = 0
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        tot_moves = []
        for move_list in moves:
            tot_moves.extend(move_list)

        if 'Dondozo' in self.member_names:
            if 'Tatsugiri-curly' in self.member_names:
                score += 5

        if 'Annihilape' in self.member_names:
            if 'beat-up' in tot_moves:
                score += 5

        if 'Archaludon' in self.member_names:
            if 'beat-up' in tot_moves:
                score += 5

        return score

    def bst_avg(self):
        bst_avg = 0
        for index, member in self.team.items():
            bst_avg += member['bst']
        return bst_avg/600

    def speed_spread(self):
        speeds = [self.team[f'Member {index}']['stats']['speed'] for index in range(1,7)]
        return float(np.mean(speeds))/100 #, float(np.std(speeds))/100]

    def def_synergy(self):
        defense = [self.team[f'Member {index}']['defensive_coverage'] for index in range(1,7)]
        def_score = 0
        for member in defense:
            imm = len(member['Immune'])
            sr = len(member['Super Resistant'])
            r = len(member['Resistant'])
            w = len(member['Weak'])
            sw = len(member['Super Weak'])
            rest = 18 - imm - sr - r - w - sw
            def_score += (0.25*sr + 0.5*r + rest + 2.0*w + 4.0*sw)/18
        return def_score/6
    
    def core_synergy(self):
        defense = [self.team[f'Member {index}']['defensive_coverage'] for index in range(1,7)]
        overlap = np.ones([6, 6])
        for i in range(6):
            member_check = self.team[f'Member {i+1}']
            weak = member_check['defensive_coverage']['Weak']
            sp_weak = member_check['defensive_coverage']['Super Weak']
            for j in range(6):
                weak_resist = [item for item in weak if item in defense[j]['Resistant']]
                weak_sp_resist = [item for item in weak if item in defense[j]['Super Resistant']]
                weak_immune = [item for item in weak if item in defense[j]['Immune']]

                sp_weak_resist = [item for item in sp_weak if item in defense[j]['Resistant']]
                sp_weak_sp_resist = [item for item in sp_weak if item in defense[j]['Super Resistant']]
                sp_weak_immune = [item for item in sp_weak if item in defense[j]['Immune']]

                overlap[i,j] = len(weak_resist) + 2*len(weak_sp_resist) + 4*len(weak_immune) + 2*len(sp_weak_resist) + 4*len(sp_weak_sp_resist) + 8*len(sp_weak_immune)

        syn = []
        for j in range(6):
            syn.append(float(sum(overlap[:,j])/5))
        
        return np.mean(syn)

    def display_team_scores(self):
        print(self.member_names)
        print("Core:", self.core_synergy())
        print("Def:", self.def_synergy())
        print("Off:", self.off_synergy())
        print("Spd:", self.speed_spread())
        print("BST Avg:", self.bst_avg())
        print("Move:", self.move_scores())
        print("Roles:", self.role_composition())
        print("Speed Control:", self.speed_control())
        print("Pivot Control:", self.pivoting_moves())
        print("Weather:", self.weather())
        print("Terrain:", self.terrain())
        print("Screen:", self.screens())
        print("Random:", self.random())

    def off_synergy(self):
        offense = [self.team[f'Member {index}']['stab_coverage'] for index in range(1,7)]
        off_score = 0
        for member in offense:
            imm = len(member['Immune'])
            r = len(member['Resistant'])
            w = len(member['Strong'])
            rest = 18 - r - w - imm
            off_score += (0.5*r + rest + 2.0*w)/18
        return off_score/6

def parse_pokepaste(url, index = 1):
    try:
        stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        other_stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

        url = f"{url}/raw"
        response = requests.get(url)
        response.raise_for_status()

        text = response.text.strip()

        # Split into mon blocks
        blocks = [b.strip() for b in text.split('\r\n\r') if b.strip()]
        team = {}

        for j, block in enumerate(blocks):
            lines = block.split('\n')

            name_part, item = lines[0].split('@')
            name_part = name_part.strip().lower()
            item = item.strip().replace(' ','-').lower()

            pattern1 = r"^(?P<nickname>.+?) \((?P<species>.+?)\)(?: \((?P<gender>[MF])\))?$"
            pattern2 = r"^(?P<species>.+?)(?: \((?P<gender>[MF])\))?$"
            pattern3 = r"^(?P<nickname>.+?) \((?P<species>.+?)\)?$"
            m = re.match(pattern1, name_part)
            if m:
                name = m.group("species")
            m = re.match(pattern2, name_part)
            if m:
                name = m.group("species")
            m = re.match(pattern3, name_part)
            if m:
                name = m.group("species")

            if ')' in name:
                name, stuff = name.split(')')
                name = name.strip()

            evs_dict = {other_stat:0 for other_stat in other_stats}
            ivs_dict = {other_stat:31 for other_stat in other_stats}

            data = {
                'name': name,
                'item': item,
                'ability': None,
                'tera_type': None,
                'evs': None,
                'nature': None,
                'ivs': None,
                'moves': []
            }

            for line in lines[1:]:
                if line.startswith('Ability:'):
                    data['ability'] = line.split(':', 1)[1].strip().lower().replace(' ','-')

                elif line.startswith('Tera Type:'):
                    data['tera_type'] = line.split(':', 1)[1].strip().lower()

                elif line.startswith('EVs:'):
                    data['evs'] = line.split(':', 1)[1].strip().lower()

                elif 'Nature' in line:
                    data['nature'] = line.strip().replace(' Nature','').lower()

                elif line.startswith('IVs:'):
                    data['ivs'] = line.split(':', 1)[1].strip().lower()

                elif line.startswith('- '):
                    data['moves'].append(line[2:].strip())

            if data['evs'] is not None:
                evs = data['evs'].split(' / ')
                for ev in evs:
                    for stat, other_stat in zip(stats, other_stats):
                        if stat in ev:
                            evs_dict[other_stat] = int(ev.replace(f" {stat}",''))

            if data['ivs'] is not None:
                ivs = data['ivs'].split(' / ')
                for iv in ivs:
                    for stat, other_stat in zip(stats, other_stats):
                        if stat in iv:
                            ivs_dict[other_stat] = float(iv.replace(f" {stat}",''))

            data['evs'] = evs_dict
            data['ivs'] = ivs_dict
            data['moves'] = [move.lower().replace(' ','-') for move in data['moves']]

            team[f"Member {j+1}"] = data
        return team

    except requests.RequestException as e:
        return f"Error fetching data: {e}. Team Number:{index}"

def pikalytics(pokemon_name):
    stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    try:
        url = f"https://www.pikalytics.com/pokedex/gen9vgc2025reghbo3/{pokemon_name}"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        container = soup.find_all('div', class_='inline-block pokemon-stat-container')

        moves = []
        partners = []
        items = []
        abilities = []
        evs = []

        stats_container = container[0].select('div[style*="display:inline-block;vertical-align: middle;margin-left: 20px;"]')
        moves_container = container[1].find_all('div', class_='pokedex-move-entry-new')
        partners_container = container[2].find_all('a', class_='teammate_entry')
        items_container = container[3].find_all('div', class_='pokedex-move-entry-new')
        abilities_container = container[4].find_all('div', class_='pokedex-move-entry-new')
        evs_container = container[5].find_all('div', class_='pokedex-move-entry-new')

        stats = [int(div.get_text(strip=True)) for div in stats_container]
        base_stats = {stat_name: stat for stat_name, stat in zip(stat_names, stats)}

        for move in moves_container:
            move_info = move.find_all('div', style=lambda value: 'inline-block' in value)
            moves.append([move_info[0].text.replace(' ', '-').lower(), float(move_info[2].text.replace('%',''))/100])
        
        for partner in partners_container:
            partner_info = partner.find_all('div', style=lambda value: 'inline-block' in value)
            partners.append([partner_info[2].text.strip().replace(' ', '-').lower(), float(partner_info[-1].text.replace('%',''))/100])
        
        for item in items_container:
            item_info = item.find_all('div', style=lambda value: 'inline-block' in value)
            items.append([item_info[2].text.replace(' ', '-').lower(), float(item_info[3].text.replace('%',''))/100])
        
        for ability in abilities_container:
            ability_info = ability.find_all('div', style=lambda value: 'inline-block' in value)
            abilities.append([ability_info[0].text.replace(' ', '-').lower(), float(ability_info[1].text.replace('%',''))/100])
        
        for ev in evs_container:
            ev_info = ev.find_all('div', style=lambda value: 'inline-block' in value)
            ev_holder = {}
            ev_holder['nature'] = ev_info[0].text.replace(' ', '-').lower()
            for i, stat in enumerate(stat_names):
                ev_holder[stat] = int(ev_info[i+1].text.strip().replace('/',''))
            ev_holder['usage'] = float(ev_info[-1].text.strip().replace('%',''))/100
            evs.append(ev_holder)

        types_container = soup.find('div', class_= 'inline-block content-div-header-font')
        types = [type.text for type in types_container.find_all('span', class_='type')]


        return {
            'name': pokemon_name,
            'types': types,
            'base_stats': base_stats,
            'moves': moves,
            'partners': partners,
            'items': items,
            'abilities': abilities,
            'natures/evs': evs
        }

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")

def construct_usage(list, cut_off):
# usage_txt = ["gen9vgc_sep2024reghbo3-1760" , "gen9vgc_oct2024reghbo3-1760" , "gen9vgc_nov2024reghbo3-1760" , "gen9vgc_dec2024reghbo3-1760" , "gen9vgc_aug2025reghbo3-1760" , "gen9vgc_sep2025reghbo3-1760", "gen9vgc_oct2025reghbo3-1760"]
# usage = construct_usage(usage_txt, 5)
#usage.to_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\usage.csv")

    df = pd.DataFrame()
    for text in list:
        file_path = f"C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\{text}.txt"
        df1 = pd.read_table(file_path, delimiter="|", skiprows = 1, header = None)
        df1 = df1.drop(columns = [0,1,4,5,6,7,8])
        df1.columns = ['Pokemon', 'Usage Percent']
        df1['Pokemon'] = df1['Pokemon'].str.strip()
        df1['Usage Percent'] = pd.to_numeric(df1['Usage Percent'].str.replace('%', ''))
        df1 = df1.sort_values('Pokemon')
        df1 = df1.set_index("Pokemon")
        df1 = df1.loc[df1["Usage Percent"] > cut_off]
        df = pd.concat([df, df1], axis = 1).fillna(0)
    df.columns = ["Sep2024 Usage %", "Oct2024 Usage %", "Nov2024 Usage %", "Dec2024 Usage %", "Aug2024 Usage %", "Sep2025 Usage %", "Oct2025 Usage %"]
    df["Average Usage"] = np.average(df, axis = 1, weights = [0.05, 0.05, 0.07, 0.07, 0.1, 0.33, 0.33])
    df = df/100
    df = df.sort_values("Average Usage", ascending = False)
    df = df.reset_index()
    return df

def clean_data():
    unfinished_moves = []
    unrecognized_pokemon = []

    for pokemon in range(len(usage)):
        pokemon_name = usage.iloc[pokemon]["Pokemon"]
        pokemon_info = get_pokemon_info(pokemon_name)
        print(f"Getting info for: {pokemon_name}")
        if pokemon_info is not None:
            for move in pokemon_info["moves"]:
                move_info = get_move_info(move)
                if move_info["flag"] == True:
                    pass
                else:
                    print(move_info["name"])
                    unfinished_moves.append(move)
        else:
            unrecognized_pokemon.append(pokemon_name)

    print("Finished")
    print(unfinished_moves)
    for pokemon in range(len(usage)):
        pokemon_name = usage.iloc[pokemon]["Pokemon"]
        poke = get_pokemon_info(pokemon_name)
        print(poke['name'])
        for move in poke['moves']:
            moves_list.add(move)
        time.sleep(0.5)

    for move in fixed_moves:
        moves_list.remove(move)

    moves_list = pd.Series(list(moves_list))
    moves_list.to_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\Not Fixed Moves.csv", index = False)

    moves_list = pd.read_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\Not Fixed Moves.csv", header = None)

def pickle_pokemon():
    for pokemon in range(len(usage)):
        pokemon_name = usage.iloc[pokemon]["Pokemon"]
        pokedex.register_pokemon(get_pokemon_info(pokemon_name))
        time.sleep(0.1)
        print(pokemon_name)

    # Save to file
    with open('pokedex.pkl', 'wb') as file:  # 'wb' = write binary
        pickle.dump(pokedex.pokemon_database, file)

move_dex = Move_Dex()
pokedex = Pokedex()
type_chart = Type_Chart()

# moves_list = set()
usage = pd.read_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\usage.csv", index_col = 0)
# fixed_moves = pd.read_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\Unfinished Moves Fixed.csv", index_col = 0)

# for move in fixed_moves.columns:
#     if pd.isna(fixed_moves[move]['stat_changes1']):
#         fixed_moves.loc['stat_changes', move] = []
#     else:
#         fixed_moves.loc['stat_changes', move] = [{'change': fixed_moves[move]['stat_changes1'], 'stat':{'name': fixed_moves[move]['stat_changes2']}},
#                                 {'change': fixed_moves[move]['stat_changes1'], 'stat':{'name': fixed_moves[move]['stat_changes3']}}]
# fixed_moves = fixed_moves.drop(['stat_changes1', 'stat_changes2', 'stat_changes3'])

# fixed_moves = fixed_moves.to_dict()

# for moves in fixed_moves:
#     move_name = moves
#     move_info = fixed_moves[move_name]
#     move_dex.register_move(move_info)

# moves_list = pd.read_csv("C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\Not Fixed Moves.csv", header = None)

# for move in moves_list[0]:
#     move_info = get_move_info(move)
#     move_dex.register_move(move_info)
#     time.sleep(0.1)
#     print(move_info)

# # Save to file
# with open('move_dex.pkl', 'wb') as file:  # 'wb' = write binary
#     pickle.dump(move_dex.move_database, file)

# Load from file
with open('move_dex.pkl', 'rb') as file:  # 'rb' = read binary
    loaded_data = pickle.load(file)

move_dex.load_moves(loaded_data)

# move_dex.choose_move_set(role = 'support', move_list = [name for name in fixed_moves], status_flag=True)

# for move in range(len(moves_list)):
#     move_name = moves_list[0][move]
#     move_info = get_move_info(move_name)
#     move_dex.register_move(move_info)
#     time.sleep(0.1)

# Load from file
with open('pokedex.pkl', 'rb') as file:  # 'rb' = read binary
    loaded_data = pickle.load(file)

pokedex.load_data(loaded_data)

name_list = []

for pokemon in range(1):#len(usage)):
    pokemon_name = usage.iloc[pokemon]["Pokemon"]
    name_list.append(pokemon_name)

#NEED TO FIX WEIRD NAMES LIKE MAUSHOLD...
for name in name_list:
    print(pikalytics(name))

teams = Teams()
#teams.create_team(name_list)
#print(teams.team)

# num_pokemon = 1

# for pokemon in range(len(usage)):
#     pokemon_name = usage.iloc[pokemon]["Pokemon"]
#     pokemon = pokedex.create_multiple_instances(pokemon_name, count = num_pokemon)
#     for poke in range(num_pokemon):
#         pokedex.print_pokemon_details(pokemon[poke])

#teams.display_team_scores()
#teams.print_team_details()

#possible changes
#-random item from list of possible used items?

# url = 'https://pokepast.es/c5f5b2d151a82b44'
# teams.load_team(parse_pokepaste(url))
# print(teams.team)

# try:
#     url = 'https://www.nimbasacitypost.com/2025/08/regulation-h-sample-teams.html'
#     response = requests.get(url)
#     response.raise_for_status()

#     soup = BeautifulSoup(response.content, 'html.parser')
#     team_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('https://pokepast.es/')]
#     for i, team in enumerate(team_links):
#         print(parse_pokepaste(team, i))
#         time.sleep(1)
# except requests.RequestException as e:
#     print(f"Error fetching data: {e}")
