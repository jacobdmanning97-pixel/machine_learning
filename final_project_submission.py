import requests
import time
import pandas as pd
import numpy as np
import random
from bs4 import BeautifulSoup
import re
import pickle
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

#Decorator for scrapping
def scrape(func):  
    def wrap(*args, **kwargs):
        if 'html' in args[0]:
            match = re.search(r'/([^/]+)\.html$', args[0])
            tournament = match.group(1)
            result =' '.join(word.title() for word in tournament.split('-'))
        else:
            result = args[0]
        print(f"Getting info for {result}")
        time.sleep(1)
        return func(*args, **kwargs)
    return wrap  

#Used if pokelytics does not have move information needed
def get_api_move_info(move_name):
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

#Stores information about the type chart
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

type_chart = Type_Chart()

#Get information about the pokemon movesets
def get_pokemon_moves(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url)

    if response.status_code == 200:
        pokemon_data = response.json()

        return [move["move"]["name"] for move in pokemon_data["moves"]]
    
    else:
        return None

#Scrapes pikalytics
@scrape
def pikalytics(pokemon_name):
    stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    try:
        name = pokemon_name.lower()
        if '-male' in name:
            name = name.replace('-male', '')
        if '-female' in name:
            name = name.replace('-female', '-f')
        if name == 'maushold-family-of-four':
            name = 'maushold'
        if name == 'tauros-paldea-aqua-breed':
            name = 'tauros-paldea-aqua'
        if name == 'tatsugiri-curly':
            name = 'tatsugiri'
        if name == 'rotom [wash rotom]':
            name = 'rotom-wash'

        url = f"https://www.pikalytics.com/pokedex/gen9vgc2025reghbo3/{name}"
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

        pokemon_info = {
            'name': pokemon_name,
            'types': types,
            'base_stats': base_stats,
            'moves': moves,
            'partners': partners,
            'items': items,
            'abilities': abilities,
            'natures/evs': evs
        }

        pokemon_info['def_coverage'] = type_chart.def_coverage(pokemon_info['types'])
        pokemon_info['stab_coverage'] = type_chart.get_stab_coverage(pokemon_info['types'])

        bst = 0
        for stat, value in pokemon_info['base_stats'].items():
            bst += value
        pokemon_info['base_stat_total'] = bst

        return pokemon_info

    except requests.RequestException as e:
        pokedex.register_pokemon({'name': pokemon_name})
        print(f"Error fetching data: {e}")

#Reads pokepastes
def parse_pokepaste(series, index = 1):
    time.sleep(0.5)
    if pd.isna(series.iloc[2]):
        placement = series.iloc[0]
        cp = series.iloc[1]
        names = [series.iloc[i] for i in range(3,9)]
        team = {}

        for i, name in enumerate(names):
            if name not in pokedex.pokemon_database:
                try:
                    pokemon = pikalytics(name)
                    pokedex.register_pokemon(pokemon)
                except:
                    pokedex.register_pokemon({'name': name})

            team[f"Member {i+1}"] = pokedex.create_pokepaste(name)
        
        team['placement'] = int(placement)
        team['cp'] = float(cp)/500

        return team

    else:
        placement = series.iloc[0]
        cp = series.iloc[1]
        url = series.iloc[2]
        names = [series.iloc[i] for i in range(3,9)]

        try:
            stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
            other_stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

            url = f"{url}/raw"
            response = requests.get(url)
            response.raise_for_status()

            text = response.text.strip()

            blocks = [b.strip() for b in text.split('\r\n\r') if b.strip()]
            team = {}

            split_names = []

            for name in names:
                split_names.append(name.split('-')[0])

            for j, block in enumerate(blocks):
                lines = block.split('\n')

                name_part, item = lines[0].split('@')
                name_part = name_part.strip().lower()
                item = item.strip().replace(' ','-').lower()

                name = next((names[i] for i,pkm in enumerate(split_names) if pkm.lower() in name_part.lower()), None)

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
            
            team['placement'] = int(placement)
            team['cp'] = float(cp)/500

            return team
        except requests.RequestException as e:
            return f"Error fetching data: {e}. Team Number:{index}"

#Not used      
def nimbasacitypost_regulation_h_sample_teams():
    details = []
    try:
        url = 'https://www.nimbasacitypost.com/2025/08/regulation-h-sample-teams.html'
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        target_rows = soup.find_all('tr', style=lambda value: value and 'height: 0pt' in value)

        for target in target_rows:
            info = target.find_all('td', style=lambda value: value and 'background-color' in value)
            img_tags = info[2].find_all('img')
            team = [img.get('alt') for img in img_tags]

            for i, member in enumerate(team):
                if 'Basculegion' in member:
                    team[i] = 'basculegion-male'
                if 'Maushold' in member:
                    team[i] = 'maushold-family-of-four'
                if '[Bloodmoon]' in member:
                    team[i] = 'ursaluna-bloodmoon'
                if '[Hisuian Form]' in member:
                    team[i] = team[i].replace(' [Hisuian Form]', '-hisui')
                if 'Indeedee' in member:
                    if '[Female]' in member:
                        team[i] = 'indeedee-female'
                    if '[Male]' in member:
                        team[i] = 'indeedee-male'
                if '[Alolan Form]' in member:
                    team[i] = team[i].replace(' [Alolan Form]', '-alola')
                if '[Galarian Form]' in member:
                    team[i] = team[i].replace(' [Galarian Form]', '-galar')
                if '[Paldean Form - Aqua Breed]' in member:
                    team[i] = 'tauros-paldea-aqua-breed'
                if 'Sinistcha' in member:
                    team[i] = 'sinistcha'
                if 'Tatsugiri' in member:
                    if '[Curly Form]' in member:
                        team[i] = 'tatsugiri-curly'
                team[i] = team[i].lower()

            try:
                details.append([info[1].get_text(strip = True), team, info[3].find('a')['href']])
            except: pass
        
        return details

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")

#Scrapping for teams
@scrape
def nimbasacity_results(url):
    details = []

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        if 'regional' or 'international' in url:
            target_rows = soup.find_all('tr', class_=lambda x: x and 'player-result' in x)

            for target in target_rows:
                info = target.find_all('td', style=lambda x: x and 'text-align' in x)
                placement = info[0].text
                cp = int(info[2].text)
                try:
                    href = info[1].find('a')['href']
                except:
                    break
                img_tags = info[4].find_all('img')
                team = [img.get('alt') for img in img_tags]

                for i, member in enumerate(team):
                    if 'Basculegion' in member:
                        team[i] = 'basculegion-male'
                    if 'Maushold' in member:
                        team[i] = 'maushold-family-of-four'
                    if '[Bloodmoon]' in member:
                        team[i] = 'ursaluna-bloodmoon'
                    if '[Hisuian Form]' in member:
                        team[i] = team[i].replace(' [Hisuian Form]', '-hisui')
                    if 'Indeedee' in member:
                        if '[Female]' in member:
                            team[i] = 'indeedee-female'
                        if '[Male]' in member:
                            team[i] = 'indeedee-male'
                    if '[Alolan Form]' in member:
                        team[i] = team[i].replace(' [Alolan Form]', '-alola')
                    if '[Galarian Form]' in member:
                        team[i] = team[i].replace(' [Galarian Form]', '-galar')
                    if '[Paldean Form - Aqua Breed]' in member:
                        team[i] = 'tauros-paldea-aqua-breed'
                    if 'Sinistcha' in member:
                        team[i] = 'sinistcha'
                    if 'Tatsugiri' in member:
                        if '[Curly Form]' in member:
                            team[i] = 'tatsugiri-curly'
                    team[i] = team[i].lower()

                try:
                    details.append([placement, cp, team, href])
                except: pass


        if 'special' or 'premier' in url:
            target_rows = soup.find_all('tr', style=lambda x: x and 'height: 0pt' in x)
            for target in target_rows[1:]:
                info = target.find_all('span', style=lambda x: x and 'font-family: inherit' in x)
                placement = info[0].text

                if placement in cp_lookup:
                    cp = cp_lookup[placement]
                else:
                    cp = 0

                try:
                    href = target.find('a')['href']
                except:
                    href = None

                team_info = target.find_all('p', dir='ltr')
                img_tags = team_info[3].find_all('img')
                team = [img.get('alt') for img in img_tags]

                for i, member in enumerate(team):
                    if 'Basculegion' in member:
                        team[i] = 'basculegion-male'
                    if 'Maushold' in member:
                        team[i] = 'maushold-family-of-four'
                    if '[Bloodmoon]' in member:
                        team[i] = 'ursaluna-bloodmoon'
                    if '[Hisuian Form]' in member:
                        team[i] = team[i].replace(' [Hisuian Form]', '-hisui')
                    if 'Indeedee' in member:
                        if '[Female]' in member:
                            team[i] = 'indeedee-female'
                        if '[Male]' in member:
                            team[i] = 'indeedee-male'
                    if '[Alolan Form]' in member:
                        team[i] = team[i].replace(' [Alolan Form]', '-alola')
                    if '[Galarian Form]' in member:
                        team[i] = team[i].replace(' [Galarian Form]', '-galar')
                    if '[Paldean Form - Aqua Breed]' in member:
                        team[i] = 'tauros-paldea-aqua-breed'
                    if 'Sinistcha' in member:
                        team[i] = 'sinistcha'
                    if 'Tatsugiri' in member:
                        if '[Curly Form]' in member:
                            team[i] = 'tatsugiri-curly'
                    team[i] = team[i].lower()

                try:
                    details.append([placement, cp, team, href])
                except: pass
        
        return details

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")

#Stores information about each pokemon
class Pokedex:
    def __init__(self):
        self.pokemon_database = {}
    
    def register_pokemon(self, poke_info):
        """Register a Pokemon in the database"""
        if poke_info['name'].lower() not in self.pokemon_database:
            self.pokemon_database[poke_info['name'].lower()] = poke_info
        else:
            print("Pokemon already in database")

    def delete_pokemon(self, pokemon_name):
        del self.pokemon_database[pokemon_name]

    def load_database(self, database):
        self.pokemon_database = database

    def load_instance(self, loaded):
        pokemon_name = loaded['name'].lower()
        if pokemon_name not in pokedex.pokemon_database:
            try:
                pokemon = pikalytics(pokemon_name)
                pokedex.register_pokemon(pokemon)
            except:
                pokedex.register_pokemon({'name': pokemon_name})

        base_data = self.pokemon_database[pokemon_name]
        base_stats = base_data['base_stats']
        ability = loaded['ability']
        item = loaded['item']
        nature = loaded['nature']
        tera = loaded['tera_type']
        evs = loaded['evs']
        ivs = loaded['ivs']
        move_names = loaded['moves']

        if tera == 'stellar':
            tera_coverage = 'Offensive'
        elif tera in base_data['types']:
            tera_coverage = 'Offensive'
        else:
            tera_coverage = type_chart.def_coverage([tera])

        ev_tot = 0
        if evs is not None:
            for stat, ev in evs.items():
                ev_tot += ev

        if evs is None or ev_tot == 0:
            #redo ivs for sp_attacker maybe speed iv too
            evs_container = base_data['natures/evs']
            if nature == None:
                nature = evs_container[0]['nature']   
            weight = [container['usage'] for container in evs_container]
            evs_container = random.choices(evs_container, weights = weight, k = 1)
            evs = dict(list(evs_container[0].items())[1:-1])

        if ivs == None:
            pre_role = self.get_nature_modifiers(nature.capitalize())
            if 'sp_attacker' in pre_role:
                if pre_role['sp_attacker'] == 1.1:
                    ivs = {'hp': 31, 'attack': 0, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}
            else:
                ivs = {'hp': 31, 'attack': 31, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}

        stats = self.calculate_final_stats(base_stats, evs, ivs, nature)
        bst = base_data['base_stat_total']
        roles = self.define_role(base_stats, bst, stats)

        return {
            'name': pokemon_name,
            'item': item,
            'ability': ability,
            'types': base_data['types'],
            'tera': tera,
            'tera_coverage': tera_coverage,
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

    def define_role(self, base_stats, bst, stats):
        roles = []

        max_atk = max(['attack', 'sp_attack'], key = lambda stat: stats[stat])
        bulk_score = (base_stats['hp'] + base_stats['defense'] + base_stats['sp_defense'])/3

        if bst/6 <= base_stats[max_atk]:
            roles.append(f'{max_atk}er')
        if bst/6 <= bulk_score:
            roles.append('bulky')
        if bst/6 <= base_stats['speed']:
            roles.append('speedy')

        return roles

    def create_pokepaste(self, pokemon_name):
        if pokemon_name not in self.pokemon_database:
            raise ValueError(f"Pokemon {pokemon_name} not found in database")
        pokemon_data = self.pokemon_database[pokemon_name]

        types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice',  'fighting',
                'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 
                'ghost', 'dragon', 'dark', 'steel', 'fairy', 'stellar']

        tera = random.choice(types)

        items_container = pokemon_data['items']
        weight = [container[-1] for container in items_container]
        item = random.choices(items_container, weights = weight, k = 1)[0][0]

        if pokemon_name != 'ursaluna-bloodmoon':
            abilities_container = pokemon_data['abilities']
            weight = [container[-1] for container in abilities_container]
            ability = random.choices(abilities_container, weights = weight, k = 1)[0][0]
        else:
            ability = "mind's-eye"

        moves_container = pokemon_data['moves']
        weight = [container[-1] for container in moves_container]
        moves = random.choices(moves_container, weights = weight, k = 4)
        chosen_indices = np.random.choice(len(moves_container), size = 4, replace = False, p = np.array(weight)/sum(weight))
        moves = [moves_container[move] for move in chosen_indices.tolist()]
        moves = [move[0] for move in moves]

        all_moves = get_pokemon_moves(pokemon_name)

        for move in moves:
            if move != 'other':
                try:
                    all_moves.remove(move)
                except:
                    pass

        for i, move in enumerate(moves):
            if move == 'other':
                moves[i] = move.replace('other', random.choice(all_moves))

        return {
            'name': pokemon_data['name'],
            'item': item,
            'ability': ability,
            'tera_type': tera,
            'evs': None,
            'nature': None,
            'ivs': None,
            'moves': moves
        }

    def create_instance(self, loaded_data):
        pokemon_name = loaded_data['name']
        pokemon_data = self.pokemon_database[pokemon_name]

        types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice',  'fighting',
                'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 
                'ghost', 'dragon', 'dark', 'steel', 'fairy', 'stellar']
        base_stats = pokemon_data['base_stats']

        tera = random.choice(types)
        if pokemon_name != 'ursaluna-bloodmoon':
            ability_container = pokemon_data['abilities']
            ability = random.choice(ability_container)[0]
        else:
            ability = "mind's-eye"

        item_container = pokemon_data['items']
        item = random.choice(item_container)[0]

        if tera == 'stellar':
            tera_coverage = 'Offensive'
        elif tera in pokemon_data['types']:
            tera_coverage = 'Offensive'
        else:
            tera_coverage = type_chart.def_coverage([tera])

        nature = self.get_random_nature()   
        bst = pokemon_data['base_stat_total']

        evs = self.generate_random_evs(nature)

        pre_role = self.get_nature_modifiers(nature.capitalize())

        if 'sp_attacker' in pre_role:
            if pre_role['sp_attacker'] == 1.1:
                ivs = {'hp': 31, 'attack': 0, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}
        else:
            ivs = {'hp': 31, 'attack': 31, 'defense': 31, 'sp_attack': 31, 'sp_defense': 31, 'speed': 31}

        stats = self.calculate_final_stats(base_stats, evs, ivs, nature)

        roles = self.define_role(base_stats, bst, stats)

        moves = move_dex.choose_move_set(roles, get_pokemon_moves(pokemon_name))
        move_names = [move['name'] for move in moves]

        return {
            'name': pokemon_data['name'],
            'item': item,
            'ability': ability,
            'types': pokemon_data['types'],
            'tera': tera,
            'tera_coverage': tera_coverage,
            'roles': roles,
            'defensive_coverage': pokemon_data['def_coverage'],
            'stab_coverage': pokemon_data['stab_coverage'],
            'nature': nature,
            'evs': evs,
            'ivs': ivs,
            'bst': bst,
            'max_stat': max(dict(list(stats.items())[1:]), key = stats.get),
            'min_stat': min(dict(list(stats.items())[1:]), key = stats.get),
            'stats': stats,
            'base_stats': base_stats,
            'moves': move_names
        }
    
    def generate_random_evs(self, nature) -> Dict[str, int]:
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

    def generate_random_evs(self, nature) -> Dict[str, int]:
        """Generate random EV spread that sums to 510 or less"""
        stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        evs = {stat: 0 for stat in stats}
        nature_stats = list(self.get_nature_modifiers(nature))
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

    def get_random_nature(self) -> str:
        """Get a random nature"""
        natures = [
            'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
            'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
            'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
            'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
            'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
        ]
        return random.choice(natures)

    def calculate_final_stats(self, base_stats: Dict[str, int], evs: Dict[str, int],  ivs: Dict[str, int],
                            nature: str) -> Dict[str, int]:
        """Calculate final stats considering base stats, EVs, IVs, nature, and level"""
        stats = {}
        nature_modifiers = self.get_nature_modifiers(nature.capitalize())
        for stat, base in base_stats.items():
            if stat == 'hp':
                stats[stat] = self.calculate_hp_stat(base, evs[stat])
            else:
                stats[stat] = self.calculate_other_stat(base, evs[stat], ivs[stat], nature_modifiers.get(stat, 1.0))
        return stats
    
    def calculate_hp_stat(self, base: int, ev: int) -> int:
        """Calculate HP stat"""
        # HP formula: ((2 * Base + IV + EV/4) * Level) / 100 + Level + 10
        return ((2 * base + 31 + ev // 4) * 50) // 100 + 50 + 10
    
    def calculate_other_stat(self, base: int, ev: int, iv: int, nature_modifier: float) -> int:
        """Calculate other stats (Attack, Defense, etc.)"""
        # Other stats formula: (((2 * Base + IV + EV/4) * 50) / 100 + 5) * Nature
        stat_value = ((2 * base + iv + ev // 4) * 50) // 100 + 5
        return int(stat_value * nature_modifier)
    
    def get_nature_modifiers(self, nature: str) -> Dict[str, float]:
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
        print(f"\nTera Type: {pokemon_instance['tera']}")
        print("Tera Coverage:")
        if pokemon_instance['tera_coverage'] == 'Offensive':
            print("  Tera is offensive")
        else:
            for type, coverage in pokemon_instance['tera_coverage'].items():
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

pokedex = Pokedex()

#Stores different information about the moves
class Move_Dex:
    def __init__(self):
        self.move_database = {}
    
    def register_move(self, move_info):
        """Register a Move in the database"""
        self.move_database[move_info['name'].lower()] = move_info

    def load_moves(self, data):
        self.move_database = data

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
        try:
            if move_info['damage_type'] != 'status':
                if move_info['type'] in poke_info['types']:
                    rate *= 1.1
                if move_info['type'] in ['poison', 'dark', 'ghost', 'water', 'ground', 'fire', 'fighting', 'bug', 'ice', 'flying', 'rock']:
                    rate *= 1.5

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
        except:
            print(move_name)
            return 1

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

    def get_move_info(self, move_name):
        if move_name not in move_dex.move_database:
            try:
                move = get_api_move_info(move_name)
                move_dex.register_move(move)
            except:
                print(f"{move_name} is not in the API")
        return self.move_database[move_name]

move_dex = Move_Dex()

#Load teams and for scoring
class Teams:
    def __init__(self, team_info, team_flag = True):
        self.team = {}
        self.member_names = []
        if team_flag == True:
            try:
                self.cp = team_info['cp']
                self.placement = team_info['placement']
            except:
                print(team_info)
            for poke in range(6):
                try:
                    self.team[f'Member {poke + 1}'] = pokedex.load_instance(team_info[f'Member {poke + 1}'])
                    self.member_names.append(team_info[f'Member {poke + 1}']['name'])
                except:
                    pass
        else:
            try:
                self.cp = team_info['cp']
                self.placement = team_info['placement']
            except:
                print(team_info)
            for poke in range(6):
                self.team[f'Member {poke + 1}'] = pokedex.create_instance(team_info[f'Member {poke + 1}'])
                self.member_names.append(team_info[f'Member {poke + 1}']['name'])

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
        return float(np.mean(move_scores))

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


        grass_improved = []
        grass_unimproved = []
        psychic_improved = []
        electric_improved = []
        mist_improved = []
        mist_unimproved = []

        items = [self.team[f'Member {index}']['item'] for index in range(1,7)]

        if 'grass' in terrain_types:
            for move in move_dict:
                if move in grass[0:3]:
                    grass_improved.append(move)
                if move in grass[3:6]:
                    grass_unimproved.append(move)
                if move_dict[move] == 'grass':
                    grass_improved.append(move)
            if 'grassy-seed' in items:
                grass_improved.append('seed')

        if 'psychic' in terrain_types:
            for move in move_dict:
                if move in psychic:
                    psychic_improved.append(move)
                if move_dict[move] == 'psychic':
                    psychic_improved.append(move)
            if 'psychic-seed' in items:
                psychic_improved.append('seed')

        if 'electric' in terrain_types:
            for move in move_dict:
                if move in electric:
                    electric_improved.append(move)
                if move_dict[move] == 'electric':
                    electric_improved.append(move)
            if 'electric-seed' in items:
                electric_improved.append('seed')

        if 'mist' in terrain_types:
            for move in move_dict:
                if move in mist:
                    mist_improved.append(move)
                if move_dict[move] == 'dragon':
                    mist_unimproved.append(move)
            if 'misty-seed' in items:
                mist_improved.append('seed')

        #print(grass_improved, grass_unimproved, psychic_improved, electric_improved, mist_improved, mist_unimproved)

        terrain_score = 0

        terrain_score += len(grass_improved)
        terrain_score -= len(grass_unimproved)
        terrain_score += len(psychic_improved)
        terrain_score += len(electric_improved)
        terrain_score += len(mist_improved)
        terrain_score -= len(mist_unimproved)

        return terrain_score

    def item_score(self):
        items = [self.team[f'Member {index}']['item'] for index in range(1,7)]
        abilities = [self.team[f'Member {index}']['ability'] for index in range(1,7)]
        score = 0

        for ability in abilities:
            if 'surge' in ability:
                for item in items:
                    if 'seed' in item:
                        if item != 'miracle-seed':
                            score += 1
                            if 'unburden' in abilities:
                                score += 1
        
        multihit_moves = ['arm-thrust', 'barrage', 'bone-rush', 'bullet-seed', 'comet-punch', 'double-slap', 'fury-attack', 'fury-swipes', 'icicle-spear', 'pin-missile', 'rock-blast', 'scale-shot', 'spike-cannon', 'tail-slap', 'water-shuriken']

        type_items = {
            'black-glasses': 'dark',
            'charcoal': 'fire',
            'chople-berry': 'fighting',
            'colbur-berry': 'dark',
            'dragon-fang': 'dragon',
            'iron-plate': 'steel',
            'metal-coat': 'steel',
            'miracle-seed': 'grass',
            'mystic-water': 'water',
            'never-melt-ice': 'ice',
            'shuca-berry': 'ground',
            'sky-plate': 'flying',
            'splash-plate': 'water',
            'spell-tag': 'ghost',
            'twisted-spoon': 'psychic'
        }

        role_items = {
            'attacker': ['choice-band', 'clear-amulet', 'life-orb', 'expert-belt', 'mirror-herb', 'power-herb', 'razor-claw', 'room-service', 'scope-lens', 'white-herb'],
            'sp_attacker': ['choice-specs', 'life-orb', 'expert-belt', 'mirror-herb', 'power-herb', 'razor-claw', 'room-service', 'scope-lens', 'throat-spray', 'white-herb'],
            'speedy': ['choice-scarf', 'covert-cloak', 'covert-cloak', 'eject-pack', 'focus-sash', 'iron-ball', 'ability-shield'],
            'bulky': ['assault-vest', 'covert-cloak', 'eject-pack', 'eviolite', 'iron-ball', 'jaboca-berry', 'leftovers', 'ability-shield', 'lum-berry', 'maranga-berry', 'mental-herb', 'power-bracer', 'rocky-helmet', 'sitrus-berry', 'weakness-policy', 'wiki-berry'],
        }

        for index in range(1,7):
            member = self.team[f'Member {index}']
            item = member['item']
            type = member['types']
            weakness = member['defensive_coverage']['Weak']
            super_weakness = member['defensive_coverage']['Super Weak']
            if item in type_items:
                if 'berry' in item:
                    if type_items[item] in weakness or type_items[item] in super_weakness:
                        score += 1
                elif type_items[item] in type:
                    score += 1
            if item in ['flame-orb', 'toxic-orb']:
                if member['ability'] == 'guts':
                    score += 1
            if item == 'loaded-dice':
                for move in member['moves']:
                    if move in multihit_moves:
                        score += 1
            for role in member['roles']:
                if item in role_items[role]:
                    score += 1

        return score

    def sleep(self):
        score = 0

        items = [self.team[f'Member {index}']['item'] for index in range(1,7)]
        abilities = [self.team[f'Member {index}']['ability'] for index in range(1,7)]
        teras = [self.team[f'Member {index}']['tera'] for index in range(1,7)]

        prevent_abilities = ['vital-spirit', 'insomnia', 'purifying-salt', 'guts', 'good-as-gold', 'poison-heal']

        tot_types = []
        for member in range(1,7):
            tot_types.extend(self.team[f'Member {member}']['types'])

        score += tot_types.count('grass')
        score += teras.count('grass')

        score = 0
        if 'safety-goggles' in items:
            score += 1

        for ability in abilities:
            if ability in prevent_abilities:
                score += 1

        return score
        
    def team_usage(self, df):
        meta_score = 0
        off_meta = 0
        for member in self.member_names:
            try:
                meta_score += df.loc[member]['Average Usage']
            except:
                off_meta += 1
        return [meta_score/6, off_meta/6]

    def screens(self):
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        items = [self.team[f'Member {index}']['item'] for index in range(1,7)]
        screen_moves = ['reflect', 'light-screen', 'aurora-veil']
        screen_count = []
        for poke in moves:
            for move in poke:
                if move in screen_moves:
                    if 'light-clay' in items:
                        screen_count.append(move)
                    screen_count.append(move)
        return len(screen_count)

    def random(self):
        score = 0
        moves = [self.team[f'Member {index}']['moves'] for index in range(1,7)]
        tot_moves = []
        for move_list in moves:
            tot_moves.extend(move_list)

        if 'dondozo' in self.member_names:
            if 'tatsugiri-curly' in self.member_names:
                score += 1

        if 'annihilape' in self.member_names:
            if 'beat-up' in tot_moves:
                score += 1

        if 'archaludon' in self.member_names:
            if 'beat-up' in tot_moves:
                score += 1

        return score

    def bst_avg(self):
        bst_avg = 0
        for index, member in self.team.items():
            bst_avg += member['bst']
        return bst_avg/600

    def speed_spread(self):
        speeds = [self.team[f'Member {index}']['stats']['speed'] for index in range(1,7)]
        return [float(np.mean(speeds))/100 , float(np.std(speeds))/100]

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
        
        return float(np.mean(syn))

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

    def team_score(self):
        if len(self.member_names) == 6:
            roles = self.role_composition()
            speed = self.speed_spread()
            team_usage = self.team_usage(usage)
            return {'core_synergy': self.core_synergy(),
                    'def_synergy': self.def_synergy(),
                    'off_synergy': self.off_synergy(),
                    'avg_speed': speed[0],
                    'std_speed': speed[1],
                    'bst_avg': self.bst_avg(),
                    'move_scores': (self.move_scores()+self.pivoting_moves()+self.screens()+self.speed_control())/4,
                    'item_scores': self.item_score(),
                    'sleep_prevention': self.sleep(),
                    'meta_usage': team_usage[0],
                    'off_meta': team_usage[1],
                    'speed_control': self.speed_control(),
                    #'pivoting_moves': self.pivoting_moves(),
                    'weather': self.weather(),
                    'terrain': self.terrain(),
                    #'screens': self.screens(),
                    'random': self.random()
            }
        else:
            return None

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

#Construct X, y for the models
def tree_data(tournament, flag):
    scores = []
    cp_result = []
    for i, team in enumerate(tournament):
        try:
            team_info = tournament[team]
            team = Teams(team_info, team_flag = flag)
            team_score = team.team_score()
            if team_score == None:
                continue
            score_vector = []
            for index, score in team_score.items():
                score_vector.append(score)
            if np.shape(scores) == (0,):
                scores = score_vector
                cp_result.append(team.cp)
            else:
                scores = np.vstack([scores, score_vector])
                cp_result.append(team.cp)
        except:
            pass

    return [scores, cp_result]

#ML Model
def random_forest_classifer(X_train, y_train, X_test, y_test):
    print(f"Size of the training data: {np.shape(X_train)}\nSize of the test data: {np.shape(X_test)}")
    rf = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=8,
        class_weight='balanced',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1
    )

    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
    y_train_cat = discretizer.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_test_cat = discretizer.transform(np.array(y_test).reshape(-1, 1)).ravel()

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(y_train_cat.astype(int)))-1))),
        ('rfe', RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 
                    n_features_to_select=10)),
        ('classifier', rf)
    ])

    pipeline.fit(X_train, y_train_cat)
    
    predictions = pipeline.predict(X_test)

    # Evaluate using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train_cat, cv=cv, scoring='accuracy')
    print("Cross-validation scores:", scores)

    mask = pipeline.named_steps['rfe'].support_

    return predictions, rf, y_test_cat, mask

# time weighted usage from smoogon
usage = pd.read_csv("Reg H Data\\usage.csv", index_col = 'Pokemon')

# load information about the moves, pokemon, and items to avoid scrapping everytime
with open('Class Pickles\\move_dex.pkl', 'rb') as file:  # 'rb' = read binary
    loaded_data = pickle.load(file)
move_dex.load_moves(loaded_data)

with open('Class Pickles\\pokedex.pkl', 'rb') as file:  # 'rb' = read binary
    loaded_data = pickle.load(file)
pokedex.load_database(loaded_data)

with open('Class Pickles\\items.pkl', 'rb') as file:  # 'rb' = read binary
    loaded_data = pickle.load(file)
all_items = loaded_data

# #load the moves that needed to be looked up
# fixed_moves = pd.read_csv("Reg H Data\\Unfinished Moves Fixed.csv", index_col = 0)

# for move in fixed_moves.columns:
#     if pd.isna(fixed_moves[move]['stat_changes1']):
#         fixed_moves.loc['stat_changes', move] = []
#     else:
#         fixed_moves.loc['stat_changes', move] = [{'change': fixed_moves[move]['stat_changes1'], 'stat':{'name': fixed_moves[move]['stat_changes2']}},
#                                 {'change': fixed_moves[move]['stat_changes1'], 'stat':{'name': fixed_moves[move]['stat_changes3']}}]
# fixed_moves = fixed_moves.drop(['stat_changes1', 'stat_changes2', 'stat_changes3'])

# fixed_moves = fixed_moves.to_dict()

# for moves in fixed_moves:
#     move_info = fixed_moves[moves]
#     move_dex.register_move(move_info)

# #want to target the most used pokemon
# w = set()
# sw = set()

# for pokemon in range(3):
#     poke = usage.loc[pokemon]['Pokemon'].lower()
#     for type in pokedex.pokemon_database[poke]['def_coverage']['Weak']:
#         w.add(type)
#     for type in pokedex.pokemon_database[poke]['def_coverage']['Super Weak']:
#         sw.add(type)

# print(w, len(w))
# print(sw, len(sw))

# #gets all the items that have been used for feature construction
# all_items = set()

# for poke in pokedex.pokemon_database:
#     try:
#         for item in pokedex.pokemon_database[poke]['items']:
#             all_items.add(item[0])
#     except:
#         pass
# all_items.remove('other')
# all_items.remove('nothing')

results = {}

#Results dictionary
cp_lookup = {'1': 350,
             '2': 325,
             '3': 300,
             '4': 300,
             '5': 280,
             '6': 280,
             '7': 280,
             '8': 280,
             '9': 160,
             '10': 160,
             '11': 160,
             '12': 160,
             '13': 160,
             '14': 160,
             '15': 160,
             '16': 160,
             '17': 125,
             '18': 125,
             '19': 125,
             '20': 125,
             '21': 125,
             '22': 125,
             '23': 125,
             '24': 125,
             '25': 125,
             '26': 125,
             '27': 125,
             '28': 125,
             '29': 125,
             '30': 125,
             '31': 125,
             '32': 125
            }

#URLs for each tournament
urls = ['https://www.nimbasacitypost.com/2024/09/baltimore-regional-2025.html', 'https://www.nimbasacitypost.com/2024/10/louisville-regional-2025.html', 'https://www.nimbasacitypost.com/2024/11/sacramento-regional-2025.html', 'https://www.nimbasacitypost.com/2024/12/toronto-regional-2025.html', 'https://www.nimbasacitypost.com/2024/09/dortmund-regional-2025.html', 'https://www.nimbasacitypost.com/2024/10/lille-regional-2025.html', 'https://www.nimbasacitypost.com/2024/11/gdansk-regional-2025.html', 'https://www.nimbasacitypost.com/2024/12/stuttgart-regional-2025.html', 'https://www.nimbasacitypost.com/2024/09/joinville-regional-2025.html', 'https://www.nimbasacitypost.com/2024/12/perth-regional-2025.html', 'https://www.nimbasacitypost.com/2024/12/bogota-special-2025.html', 'https://www.nimbasacitypost.com/2024/11/buenos-aires-special-2025.html', 'https://www.nimbasacitypost.com/2024/10/thailand-premier-ball-league-2025.html', 'https://www.nimbasacitypost.com/2024/11/singapore-premier-ball-league-2025.html', 'https://www.nimbasacitypost.com/2024/10/philippines-premier-ball-league-2025.html', 'https://www.nimbasacitypost.com/2024/12/taiwan-premier-ball-league-2025.html', 'https://www.nimbasacitypost.com/2024/11/latin-america-international-2025.html', 'https://www.nimbasacitypost.com/2025/10/belo-horizonte-regional-2026.html', 'https://www.nimbasacitypost.com/2025/09/pittsburgh-regional-2026.html', 'https://www.nimbasacitypost.com/2025/10/milwaukee-regional-2026.html', 'https://www.nimbasacitypost.com/2025/10/lille-regional-2026.html', 'https://www.nimbasacitypost.com/2025/09/monterrey-regional-2026.html', 'https://www.nimbasacitypost.com/2025/09/frankfurt-regional-2026.html', 'https://www.nimbasacitypost.com/2024/10/lima-special-2025.html']

test_urls = ['https://www.nimbasacitypost.com/2025/11/latin-america-international-2026.html']

#Gets the names the names of the tournaments
tournaments = []
test_tournaments = []
tournament_results = {}

for url in urls:
    match = re.search(r'/([^/]+)\.html$', url)
    tournament_str = match.group(1)
    tournament_name =' '.join(word.title() for word in tournament_str.split('-'))
    tournaments.append(tournament_name)
    # results[tournament_name] = nimbasacity_results(url)

for url in test_urls:
    match = re.search(r'/([^/]+)\.html$', url)
    tournament_str = match.group(1)
    tournament_name =' '.join(word.title() for word in tournament_str.split('-'))
    test_tournaments.append(tournament_name)
    # results[tournament_name] = nimbasacity_results(url)

# #gets info from the tournaments then saves them
# for tournament in tournaments:
#     tournament_results[tournament] = pd.read_csv(f"Reg H Data\\{tournament}.csv")

# for tournament in tournament_results:
#     tournament_teams = {}
#     print('=====', tournament, '=====')
#     for team in range(len(tournament_results[tournament])):
#         if team%50 == 0:
#             print(tournament, team)
#         try:
#             tournament_teams[f"{tournament} Team {team+1}"] = parse_pokepaste(tournament_results[tournament].iloc[team])
#         except:
#             print('===',tournament, team+1,'===')

#     with open(f"{tournament}.pkl", 'wb') as file:
#         pickle.dump(tournament_teams, file)

#Opens and loads information from the tournaments to run faster
tournament_teams = {}
for tournament in tournaments:
    with open(f"Tournament Pickles\\{tournament}.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    tournament_teams[tournament] = loaded_data

test_teams = {}
for tournament in test_tournaments:
    with open(f"Tournament Pickles\\{tournament}.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    test_teams[tournament] = loaded_data

#Builds X_train and y_train
score_matrix = []
for i, tournament in enumerate(tournaments):
    score_vecs, cp_vector = tree_data(tournament_teams[tournament], True)
    if i == 0:
        score_matrix = score_vecs
        cp_vectors = cp_vector
    else:
        score_matrix = np.vstack([score_matrix, score_vecs])
        cp_vectors = cp_vectors + cp_vector

#Builds X_test and y_test
for tournament in test_tournaments:
    test_score_vecs, test_cp_vector = tree_data(test_teams[tournament], True)

#Calls the model and reports
prediction, rf, y_bins, mask = random_forest_classifer(score_matrix, cp_vectors, test_score_vecs, test_cp_vector)
print(f"================Classification Report================ \n{classification_report(y_bins, prediction, target_names=[f"Class {i}" for i in range(4)])}")
feature_importance = rf.feature_importances_
feature_list = ['core_synergy', 'def_synergy', 'off_synergy', 'avg_speed', 'std_speed', 'bst_avg', 'move_scores', 'item_scores', 'sleep_prevention', 'meta_usage', 'off_meta', 'weather', 'terrain', 'random']
feature_list = [feature for feature, flag in zip(feature_list, mask) if flag]

feature_dict = {feature: float(val) for feature, val in zip(feature_list, feature_importance)}

feature_dict = {feature: val for feature, val in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)}

print("========Features========")
for feature, val in feature_dict.items():
    print(f" {feature}: {100*val:.2f}%")

cm = confusion_matrix(y_bins, prediction)
print("===Confusion Matrix===")
for row in cm:
    print("  ".join(f"{num:2d}" for num in row))

# #Used to see which pokepastes are repeated from another source
# df = pd.read_csv("Reg H Data\\VGCPastes Repository.csv", header = 0)
# df = df[df['EVs'] == 'Yes']
# df = df[df['Category'] == 'In Person Event']
# df['Rank'] = df['Rank'].apply(lambda x: ''.join(filter(str.isdigit, x)))

# for tournament in tournaments:
#     try:
#         df_temp = df[df['Tournament / Event'] == tournament]
#         df_temp = df_temp.reset_index()

#         header = ['Rank', 'CP', 'Team', 'Pokepaste']
#         results_container = pd.DataFrame(results[tournament], columns = header)

#         mask = results_container['Rank'].isin(df_temp['Rank'])
#         indices = results_container[mask].index.to_list()

#         rest = len(df_temp['Pokepaste']) - len(indices)

#         results_container.loc[mask, 'Pokepaste'] = pd.Series(df_temp['Pokepaste'][:len(indices)].to_list(), index = indices)

#         team_column = pd.DataFrame(results_container['Team'].tolist(), index = results_container.index)
#         team_column.columns = [f"Pokemon {i+1}" for i in range(team_column.shape[1])]
#         results_container = pd.concat([results_container.drop('Team', axis=1), team_column], axis=1)

#         for i in range(rest):
#             rest_info =[df_temp.loc[rest:,'Rank'][i+1], cp_lookup[df_temp.loc[rest:,'Rank'][i+1]], df_temp.loc[rest:,'Pokepaste'][i+1]]
#             rest_team = [df_temp.loc[rest:,f"Pokemon {j+1}"][i+1].lower() for j in range(6)]
#             results_container.loc[len(results_container)] = rest_info + rest_team

#         tournament_results[tournament] = results_container    

#     except:
#         print(tournament)

# #Save the information into csv files
# for name, df in tournament_results.items():
#     df.to_csv(f"C:\\Users\\jacob\\OneDrive\\Desktop\\Reg H Data\\{name}.csv", index = False)

#     teams = parse_pokepaste(url, pokemon_names, placement)

#     for member in list(teams)[:6]:
#         print(member)
#         if teams[member]['name'] != None:
#             if teams[member]['name'].lower() not in pokedex.pokemon_database:
#                 team_member = pikalytics(teams[member]['name'])
#                 if team_member != None:
#                     pokedex.register_pokemon(team_member)
#             else:
#                 team_member = pokedex.pokemon_database[teams[member]['name'].lower()]
#             if team_member != None:
#                 member_instance = pokedex.load_instance(teams[member])
#                 #pokedex.print_pokemon_details(member_instance)
#                 time.sleep(1)
#             else: 
#                 print(teams[member]['name'])
#         else: 
#             print(teams[member])
#     print(i)
