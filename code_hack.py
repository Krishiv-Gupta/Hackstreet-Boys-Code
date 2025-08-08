import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
import itertools
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class EnhancedCricketMatchSimulator:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10),
            'gradient_boost': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42, kernel='rbf')
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.player_data = None
        self.teams_data = None
        
    def load_player_statistics(self, csv_file_path="IPL 2025 Player Statistics Clean.csv"):
        """Load player statistics from the main CSV file"""
        try:
            print(f"Loading player statistics from {csv_file_path}...")
            
            # Check if file exists
            if not os.path.exists(csv_file_path):
                print(f"File {csv_file_path} not found!")
                return None
                
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print("Player statistics file is empty!")
                return None
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle missing values more robustly
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Fill non-numeric columns with empty strings
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            df[non_numeric_columns] = df[non_numeric_columns].fillna('')
            
            self.player_data = df
            print(f"Successfully loaded {len(df)} players from statistics file")
            print(f"Columns available: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: File {csv_file_path} not found!")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: {csv_file_path} is empty!")
            return None
        except Exception as e:
            print(f"Error loading player statistics: {str(e)}")
            return None
    
    def load_teams_data(self, csv_file_path):
        """Load teams data from CSV file"""
        try:
            print(f"Loading teams data from {csv_file_path}...")
            
            # Check if file exists
            if not os.path.exists(csv_file_path):
                print(f"File {csv_file_path} not found!")
                return None
                
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print("Teams data file is empty!")
                return None
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            self.teams_data = df
            print(f"Successfully loaded teams data with {len(df)} entries")
            print(f"Columns available: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: File {csv_file_path} not found!")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: {csv_file_path} is empty!")
            return None
        except Exception as e:
            print(f"Error loading teams data: {str(e)}")
            return None
    
    def extract_teams_from_data(self):
        """Extract unique teams and their players from the teams data"""
        if self.teams_data is None:
            print("No teams data loaded!")
            return None
        
        teams = {}
        
        try:
            # Check if we have the 2-column format (Player_Team1, Player_Team2)
            if 'Player_Team1' in self.teams_data.columns and 'Player_Team2' in self.teams_data.columns:
                print("Detected 2-team format (Player_Team1, Player_Team2)")
                
                # Extract players for Team 1
                team1_players = []
                for _, row in self.teams_data.iterrows():
                    player = str(row['Player_Team1']).strip()
                    if player and player.lower() not in ['nan', '', 'none']:
                        team1_players.append(player)
                
                # Extract players for Team 2
                team2_players = []
                for _, row in self.teams_data.iterrows():
                    player = str(row['Player_Team2']).strip()
                    if player and player.lower() not in ['nan', '', 'none']:
                        team2_players.append(player)
                
                teams = {
                    'Team 1': team1_players,
                    'Team 2': team2_players
                }
                
            else:
                # Try different possible column names for team and player (original format)
                team_columns = ['Team', 'team', 'Team_Name', 'TeamName', 'TEAM']
                player_columns = ['Player', 'player', 'Player_Name', 'PlayerName', 'PLAYER', 'Name', 'name']
                
                team_col = None
                player_col = None
                
                # Find the correct column names
                for col in team_columns:
                    if col in self.teams_data.columns:
                        team_col = col
                        break
                
                for col in player_columns:
                    if col in self.teams_data.columns:
                        player_col = col
                        break
                
                if team_col is None or player_col is None:
                    print("Could not find team or player columns. Available columns:")
                    print(list(self.teams_data.columns))
                    return None
                
                print(f"Using Team column: {team_col}, Player column: {player_col}")
                
                # Group players by team
                for _, row in self.teams_data.iterrows():
                    team_name = str(row[team_col]).strip()
                    player_name = str(row[player_col]).strip()
                    
                    # Skip invalid entries
                    if team_name.lower() in ['nan', '', 'none'] or player_name.lower() in ['nan', '', 'none']:
                        continue
                    
                    if team_name not in teams:
                        teams[team_name] = []
                    
                    if player_name not in teams[team_name]:  # Avoid duplicates
                        teams[team_name].append(player_name)
            
            # Remove teams with insufficient players (less than 11)
            valid_teams = {team: players for team, players in teams.items() if len(players) >= 11}
            
            if not valid_teams:
                print("No teams found with sufficient players (minimum 11)!")
                print("Team sizes found:")
                for team, players in teams.items():
                    print(f"  {team}: {len(players)} players")
                return None
            
            print(f"\nExtracted {len(valid_teams)} valid teams:")
            for team, players in valid_teams.items():
                print(f"  {team}: {len(players)} players")
            
            return valid_teams
            
        except Exception as e:
            print(f"Error extracting teams from data: {str(e)}")
            return None
    
    def get_player_stats(self, player_name):
        """Get statistics for a specific player"""
        if self.player_data is None:
            return None
        
        # Try to find player (case-insensitive)
        player_name_lower = str(player_name).lower().strip()
        
        # Check different possible player name columns
        name_columns = ['Player_Name', 'Player', 'Name', 'player', 'name', 'PLAYER']
        
        try:
            for col in name_columns:
                if col in self.player_data.columns:
                    # Convert to string and handle NaN values properly
                    col_data = self.player_data[col].astype(str)
                    col_data = col_data.str.replace('nan', '', case=False)
                    col_data = col_data.str.replace('NaN', '', case=False)
                    col_data = col_data.str.replace('No stats', '', case=False)  # Handle "No stats" values
                    mask = col_data.str.lower().str.strip() == player_name_lower
                    matches = self.player_data[mask]
                    if len(matches) > 0:
                        return matches.iloc[0].to_dict()
            
            # If exact match not found, try partial match
            for col in name_columns:
                if col in self.player_data.columns:
                    # Convert to string and handle NaN values properly
                    col_data = self.player_data[col].astype(str)
                    col_data = col_data.str.replace('nan', '', case=False)
                    col_data = col_data.str.replace('NaN', '', case=False)
                    col_data = col_data.str.replace('No stats', '', case=False)  # Handle "No stats" values
                    mask = col_data.str.lower().str.contains(player_name_lower, na=False)
                    matches = self.player_data[mask]
                    if len(matches) > 0:
                        print(f"Partial match found for {player_name}: {matches.iloc[0][col]}")
                        return matches.iloc[0].to_dict()
        
        except Exception as e:
            print(f"Error finding player {player_name}: {str(e)}")
        
        return None
    
    def calculate_player_score(self, player_stats):
        """Calculate comprehensive player performance score with improved error handling"""
        if player_stats is None:
            return 1  # Return minimum score instead of 0
        
        batting_score = 0
        bowling_score = 0
        fielding_score = 0
        
        # Helper function to safely get numeric value
        def safe_get(key, default=0):
            try:
                value = player_stats.get(key, default)
                # Handle "No stats" and other non-numeric values
                if isinstance(value, str) and value.lower() in ['no stats', 'nan', '', 'none']:
                    return default
                if pd.isna(value):
                    return default
                return float(value) if value != '' else default
            except (ValueError, TypeError):
                return default
        
        # Batting metrics (try different possible column names)
        batting_keys = {
            'avg': ['Batting_Average', 'Bat_Avg', 'Average', 'Avg'],
            'strike_rate': ['Batting_Strike_Rate', 'Strike_Rate', 'SR', 'Bat_SR'],
            'runs': ['Runs_Scored', 'Runs', 'Total_Runs'],
            'matches': ['Matches_Batted', 'Matches', 'Mat'],
            'centuries': ['Centuries', '100s', 'Hundreds'],
            'fifties': ['Half_Centuries', '50s', 'Fifties']
        }
        
        batting_avg = 0
        strike_rate = 0
        runs = 0
        centuries = 0
        fifties = 0
        
        for stat, possible_keys in batting_keys.items():
            for key in possible_keys:
                if key in player_stats:
                    if stat == 'avg':
                        batting_avg = safe_get(key)
                    elif stat == 'strike_rate':
                        strike_rate = safe_get(key)
                    elif stat == 'runs':
                        runs = safe_get(key)
                    elif stat == 'centuries':
                        centuries = safe_get(key)
                    elif stat == 'fifties':
                        fifties = safe_get(key)
                    break
        
        batting_score = (
            batting_avg * 0.3 +
            strike_rate * 0.2 +
            centuries * 10 +
            fifties * 5 +
            (runs / 100) * 0.1  # Normalize runs
        )
        
        # Bowling metrics
        bowling_keys = {
            'bowling_avg': ['Bowling_Average', 'Bowl_Avg', 'Bowling_Avg'],
            'economy': ['Economy_Rate', 'Economy', 'Econ'],
            'wickets': ['Wickets_Taken', 'Wickets', 'Wkts'],
            'strike_rate': ['Bowling_Strike_Rate', 'Bowl_SR', 'Bowling_SR']
        }
        
        bowling_avg = 50  # Default high bowling average
        economy = 8      # Default high economy
        wickets = 0
        
        for stat, possible_keys in bowling_keys.items():
            for key in possible_keys:
                if key in player_stats:
                    if stat == 'bowling_avg':
                        bowling_avg = safe_get(key, 50)
                    elif stat == 'economy':
                        economy = safe_get(key, 8)
                    elif stat == 'wickets':
                        wickets = safe_get(key)
                    break
        
        bowling_score = (
            max(0, (50 - bowling_avg)) * 0.3 +
            max(0, (10 - economy)) * 0.2 +
            wickets * 2
        )
        
        # Fielding score
        catches_keys = ['Catches_Taken', 'Catches', 'Ct']
        stumpings_keys = ['Stumpings', 'St']
        
        catches = 0
        stumpings = 0
        
        for key in catches_keys:
            if key in player_stats:
                catches = safe_get(key)
                break
        
        for key in stumpings_keys:
            if key in player_stats:
                stumpings = safe_get(key)
                break
        
        fielding_score = catches * 2 + stumpings * 3
        
        total_score = batting_score + bowling_score + fielding_score
        return max(total_score, 1)  # Minimum score of 1
    
    def calculate_team_strength(self, team_players):
        """Calculate team strength based on player statistics"""
        if team_players is None or len(team_players) == 0:
            return [1] * 15  # Return default values with minimum 1
        
        player_scores = []
        batting_avgs = []
        bowling_avgs = []
        strike_rates = []
        economies = []
        
        total_runs = 0
        total_wickets = 0
        total_catches = 0
        total_centuries = 0
        total_fifties = 0
        
        for player_name in team_players[:11]:  # Take only first 11 players
            stats = self.get_player_stats(player_name)
            if stats:
                score = self.calculate_player_score(stats)
                player_scores.append(score)
                
                # Extract specific stats with safe gets
                def safe_get(key, default=0):
                    try:
                        value = stats.get(key, default)
                        # Handle "No stats" and other non-numeric values
                        if isinstance(value, str) and value.lower() in ['no stats', 'nan', '', 'none']:
                            return default
                        if pd.isna(value):
                            return default
                        return float(value) if value != '' else default
                    except (ValueError, TypeError):
                        return default
                
                # Try different column name variations
                batting_avg = 20  # Default batting average
                for key in ['Batting_Average', 'Bat_Avg', 'Average', 'Avg']:
                    if key in stats:
                        avg_val = safe_get(key, 20)
                        if avg_val > 0:  # Only use positive values
                            batting_avg = avg_val
                        break
                
                bowling_avg = 40  # Default bowling average
                for key in ['Bowling_Average', 'Bowl_Avg', 'Bowling_Avg']:
                    if key in stats:
                        bowl_avg = safe_get(key, 40)
                        if bowl_avg > 0:  # Only use positive values
                            bowling_avg = bowl_avg
                        break
                
                strike_rate = 100  # Default strike rate
                for key in ['Batting_Strike_Rate', 'Strike_Rate', 'SR', 'Bat_SR']:
                    if key in stats:
                        sr_val = safe_get(key, 100)
                        if sr_val > 0:  # Only use positive values
                            strike_rate = sr_val
                        break
                
                economy = 7  # Default economy
                for key in ['Economy_Rate', 'Economy', 'Econ']:
                    if key in stats:
                        eco_val = safe_get(key, 7)
                        if eco_val > 0:  # Only use positive values
                            economy = eco_val
                        break
                
                batting_avgs.append(batting_avg)
                bowling_avgs.append(bowling_avg)
                strike_rates.append(strike_rate)
                economies.append(economy)
                
                # Aggregate stats
                for key in ['Runs_Scored', 'Runs', 'Total_Runs']:
                    if key in stats:
                        total_runs += safe_get(key)
                        break
                
                for key in ['Wickets_Taken', 'Wickets', 'Wkts']:
                    if key in stats:
                        total_wickets += safe_get(key)
                        break
                
                for key in ['Catches_Taken', 'Catches', 'Ct']:
                    if key in stats:
                        total_catches += safe_get(key)
                        break
                
                for key in ['Centuries', '100s', 'Hundreds']:
                    if key in stats:
                        total_centuries += safe_get(key)
                        break
                
                for key in ['Half_Centuries', '50s', 'Fifties']:
                    if key in stats:
                        total_fifties += safe_get(key)
                        break
            else:
                print(f"Warning: Stats not found for player {player_name}")
                player_scores.append(10)  # Default score for missing players
                batting_avgs.append(20)
                bowling_avgs.append(40)
                strike_rates.append(100)
                economies.append(7)
        
        # Ensure we have enough data points
        while len(player_scores) < 11:
            player_scores.append(10)
            batting_avgs.append(20)
            bowling_avgs.append(40)
            strike_rates.append(100)
            economies.append(7)
        
        # Calculate team features with safe operations
        try:
            team_features = [
                max(total_runs, 1),
                max(total_wickets, 1),
                max(np.mean(batting_avgs) if batting_avgs else 20, 1),
                max(np.mean([avg for avg in bowling_avgs if avg > 0]) if bowling_avgs else 40, 1),
                max(np.mean(strike_rates) if strike_rates else 100, 1),
                max(np.mean(economies) if economies else 7, 1),
                max(total_centuries, 0),
                max(total_fifties, 0),
                max(total_catches, 0),
                max(len([w for w in [total_wickets] if w >= 5]), 0),  # Five wicket hauls approximation
                max(len([w for w in [total_wickets] if w >= 4]), 0),  # Four wicket hauls approximation
                max(np.mean(player_scores) if player_scores else 10, 1),
                max(np.mean(sorted(batting_avgs, reverse=True)[:3]) if len(batting_avgs) >= 3 else np.mean(batting_avgs), 1),
                max(np.mean(sorted([avg for avg in bowling_avgs if avg > 0])[:3]) if len(bowling_avgs) >= 3 else np.mean(bowling_avgs), 1),
                max(np.mean(player_scores) if player_scores else 10, 1)  # Experience factor approximation
            ]
        except Exception as e:
            print(f"Error calculating team features: {str(e)}")
            team_features = [10] * 15  # Return default values
        
        return team_features
    
    def generate_training_data_from_stats(self, n_samples=2000):
        """Generate training data based on actual player statistics"""
        if self.player_data is None or len(self.player_data) == 0:
            print("No player data loaded. Using synthetic data...")
            return self.generate_synthetic_training_data(n_samples)
        
        print("Generating training data from real player statistics...")
        np.random.seed(42)
        
        training_features = []
        training_labels = []
        
        # Get all players and their scores
        all_players = []
        try:
            for _, player in self.player_data.iterrows():
                player_dict = player.to_dict()
                score = self.calculate_player_score(player_dict)
                all_players.append((player_dict, score))
            
            if len(all_players) < 22:  # Need at least 22 players for two teams
                print("Not enough players for training data generation. Using synthetic data...")
                return self.generate_synthetic_training_data(n_samples)
            
            # Sort players by score for better team selection
            all_players.sort(key=lambda x: x[1], reverse=True)
            
            for _ in range(n_samples):
                # Select random teams of 11 players each
                team1_indices = np.random.choice(len(all_players), 11, replace=False)
                remaining_indices = [i for i in range(len(all_players)) if i not in team1_indices]
                team2_indices = np.random.choice(remaining_indices, min(11, len(remaining_indices)), replace=False)
                
                team1_players = [all_players[i][0] for i in team1_indices]
                team2_players = [all_players[i][0] for i in team2_indices]
                
                # Calculate team strengths
                team1_strength = np.mean([self.calculate_player_score(p) for p in team1_players])
                team2_strength = np.mean([self.calculate_player_score(p) for p in team2_players])
                
                # Create feature vectors (simplified)
                team1_features = [
                    sum(self.calculate_player_score(p) for p in team1_players),
                    team1_strength,
                    max(np.random.normal(team1_strength * 0.3, 5), 1),  # batting avg
                    max(np.random.normal(50 - team1_strength * 0.2, 8), 1),  # bowling avg
                    max(np.random.normal(team1_strength * 1.2, 20), 1),  # strike rate
                    max(np.random.normal(8 - team1_strength * 0.05, 1.5), 1),  # economy
                ] + [max(np.random.normal(team1_strength * 0.1, 2), 0) for _ in range(9)]  # other features
                
                team2_features = [
                    sum(self.calculate_player_score(p) for p in team2_players),
                    team2_strength,
                    max(np.random.normal(team2_strength * 0.3, 5), 1),
                    max(np.random.normal(50 - team2_strength * 0.2, 8), 1),
                    max(np.random.normal(team2_strength * 1.2, 20), 1),
                    max(np.random.normal(8 - team2_strength * 0.05, 1.5), 1),
                ] + [max(np.random.normal(team2_strength * 0.1, 2), 0) for _ in range(9)]
                
                match_features = team1_features + team2_features
                
                # Determine winner based on team strengths with randomness
                win_probability = team1_strength / (team1_strength + team2_strength + 1e-8)
                winner = 1 if np.random.random() < win_probability else 0
                
                training_features.append(match_features)
                training_labels.append(winner)
        
        except Exception as e:
            print(f"Error generating training data from stats: {str(e)}")
            return self.generate_synthetic_training_data(n_samples)
        
        return np.array(training_features), np.array(training_labels)
    
    def generate_synthetic_training_data(self, n_samples=2000):
        """Generate synthetic training data (fallback method)"""
        print("Generating synthetic training data...")
        np.random.seed(42)
        
        training_features = []
        training_labels = []
        
        for _ in range(n_samples):
            team1_strength = np.random.uniform(0.3, 1.0)
            team2_strength = np.random.uniform(0.3, 1.0)
            
            # Generate team features with minimum values
            team1_features = [
                max(np.random.poisson(2000 * team1_strength), 1),
                max(np.random.poisson(80 * team1_strength), 1),
                max(np.random.normal(35 * team1_strength, 10), 1),
                max(np.random.normal(35 / team1_strength, 8), 1),
                max(np.random.normal(120 * team1_strength, 20), 1),
                max(np.random.normal(7 / team1_strength, 1.5), 1),
                max(np.random.poisson(5 * team1_strength), 0),
                max(np.random.poisson(15 * team1_strength), 0),
                max(np.random.poisson(25 * team1_strength), 0),
                max(np.random.poisson(3 * team1_strength), 0),
                max(np.random.poisson(8 * team1_strength), 0),
                max(np.random.normal(100 * team1_strength, 30), 1),
                max(np.random.normal(40 * team1_strength, 12), 1),
                max(np.random.normal(30 / team1_strength, 8), 1),
                max(np.random.normal(50 * team1_strength, 20), 1)
            ]
            
            team2_features = [
                max(np.random.poisson(2000 * team2_strength), 1),
                max(np.random.poisson(80 * team2_strength), 1),
                max(np.random.normal(35 * team2_strength, 10), 1),
                max(np.random.normal(35 / team2_strength, 8), 1),
                max(np.random.normal(120 * team2_strength, 20), 1),
                max(np.random.normal(7 / team2_strength, 1.5), 1),
                max(np.random.poisson(5 * team2_strength), 0),
                max(np.random.poisson(15 * team2_strength), 0),
                max(np.random.poisson(25 * team2_strength), 0),
                max(np.random.poisson(3 * team2_strength), 0),
                max(np.random.poisson(8 * team2_strength), 0),
                max(np.random.normal(100 * team2_strength, 30), 1),
                max(np.random.normal(40 * team2_strength, 12), 1),
                max(np.random.normal(30 / team2_strength, 8), 1),
                max(np.random.normal(50 * team2_strength, 20), 1)
            ]
            
            match_features = team1_features + team2_features
            win_probability = team1_strength / (team1_strength + team2_strength)
            winner = 1 if np.random.random() < win_probability else 0
            
            training_features.append(match_features)
            training_labels.append(winner)
        
        return np.array(training_features), np.array(training_labels)
    
    def train_models(self):
        """Train all ML models"""
        print("Training models...")
        
        try:
            # Generate training data
            X, y = self.generate_training_data_from_stats(3000)
            
            if X.shape[0] == 0:
                print("No training data generated!")
                return None
            
            # Handle any infinite or NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    print(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_scores[name] = accuracy
                    print(f"{name} accuracy: {accuracy:.3f}")
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    model_scores[name] = 0.0
            
            self.is_trained = True
            print("Model training completed!")
            return model_scores
            
        except Exception as e:
            print(f"Error in training models: {str(e)}")
            return None
    
    def predict_match(self, team1_players, team2_players):
        """Predict match outcome between two teams"""
        if not self.is_trained:
            print("Training models...")
            train_result = self.train_models()
            if train_result is None:
                print("Failed to train models!")
                return {
                    'winner': 'Team 1',  # Default prediction
                    'confidence': 0.5,
                    'team1_win_probability': 0.5,
                    'team2_win_probability': 0.5,
                    'individual_predictions': {},
                    'model_probabilities': {},
                    'team1_strength': 10,
                    'team2_strength': 10
                }
        
        try:
            # Calculate team features
            team1_features = self.calculate_team_strength(team1_players)
            team2_features = self.calculate_team_strength(team2_players)
            
            # Combine features
            match_features = np.array([team1_features + team2_features])
            
            # Handle any infinite or NaN values
            match_features = np.nan_to_num(match_features, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            match_features_scaled = self.scaler.transform(match_features)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(match_features_scaled)[0]
                    prob = model.predict_proba(match_features_scaled)[0]
                    predictions[name] = pred
                    probabilities[name] = prob
                except Exception as e:
                    print(f"Error predicting with {name}: {str(e)}")
                    predictions[name] = 0
                    probabilities[name] = [0.5, 0.5]
            
            # Ensemble prediction
            if predictions:
                ensemble_pred = 1 if sum(predictions.values()) > len(predictions) / 2 else 0
                
                # Average probabilities
                valid_probs = [prob for prob in probabilities.values() if len(prob) == 2]
                if valid_probs:
                    avg_prob_team1 = np.mean([prob[1] for prob in valid_probs])
                    avg_prob_team2 = 1 - avg_prob_team1
                else:
                    avg_prob_team1 = 0.5
                    avg_prob_team2 = 0.5
            else:
                ensemble_pred = 0
                avg_prob_team1 = 0.5
                avg_prob_team2 = 0.5
            
            return {
                'winner': 'Team 1' if ensemble_pred == 1 else 'Team 2',
                'confidence': max(avg_prob_team1, avg_prob_team2),
                'team1_win_probability': avg_prob_team1,
                'team2_win_probability': avg_prob_team2,
                'individual_predictions': predictions,
                'model_probabilities': probabilities,
                'team1_strength': np.mean(team1_features) if team1_features else 10,
                'team2_strength': np.mean(team2_features) if team2_features else 10
            }
            
        except Exception as e:
            print(f"Error in match prediction: {str(e)}")
            return {
                'winner': 'Team 1',  # Default prediction
                'confidence': 0.5,
                'team1_win_probability': 0.5,
                'team2_win_probability': 0.5,
                'individual_predictions': {},
                'model_probabilities': {},
                'team1_strength': 10,
                'team2_strength': 10
            }
    
    def run_tournament(self, teams_csv_path, statistics_csv_path="IPL 2025 Player Statistics Clean.csv"):
        """Run a complete tournament with all teams playing against each other"""
        
        try:
            print("="*80)
            print("CRICKET MATCH TOURNAMENT SIMULATOR")
            print("="*80)
            
            # Load data - explicitly check for None returns
            player_stats = self.load_player_statistics(statistics_csv_path)
            if player_stats is None or player_stats.empty:
                print("Failed to load player statistics!")
                return None
                
            teams_data = self.load_teams_data(teams_csv_path)
            if teams_data is None or teams_data.empty:
                print("Failed to load teams data!")
                return None
            
            teams = self.extract_teams_from_data()
            if teams is None or len(teams) == 0:
                print("No valid teams found!")
                return None
            
            if len(teams) < 2:
                print("Need at least 2 teams for a tournament!")
                return None
            
            # Train models
            train_result = self.train_models()
            if train_result is None:
                print("Failed to train models! Continuing with limited functionality...")
            
            # Generate all possible match combinations
            team_names = list(teams.keys())
            match_combinations = list(itertools.combinations(team_names, 2))
            
            print(f"\nStarting tournament with {len(team_names)} teams")
            print(f"Total matches to simulate: {len(match_combinations)}")
            print("="*80)
            
            # Results storage
            results = []
            team_stats = {team: {'wins': 0, 'losses': 0, 'points': 0} for team in team_names}
            
            # Run all matches
            for i, (team1_name, team2_name) in enumerate(match_combinations, 1):
                print(f"\nMatch {i}/{len(match_combinations)}: {team1_name} vs {team2_name}")
                print("-" * 60)
                
                team1_players = teams[team1_name]
                team2_players = teams[team2_name]
                
                # Predict match
                result = self.predict_match(team1_players, team2_players)
                
                # Determine actual winner names
                if result['winner'] == 'Team 1':
                    winner_name = team1_name
                    loser_name = team2_name
                else:
                    winner_name = team2_name
                    loser_name = team1_name
                
                # Update stats
                team_stats[winner_name]['wins'] += 1
                team_stats[winner_name]['points'] += 2
                team_stats[loser_name]['losses'] += 1
                
                # Store detailed result
                match_result = {
                    'match_number': i,
                    'team1': team1_name,
                    'team2': team2_name,
                    'winner': winner_name,
                    'loser': loser_name,
                    'confidence': result['confidence'],
                    'team1_win_probability': result['team1_win_probability'],
                    'team2_win_probability': result['team2_win_probability'],
                    'team1_strength': result['team1_strength'],
                    'team2_strength': result['team2_strength']
                }
                results.append(match_result)
                
                # Print match summary
                print(f"Winner: {winner_name}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"{team1_name} win probability: {result['team1_win_probability']:.2%}")
                print(f"{team2_name} win probability: {result['team2_win_probability']:.2%}")
            
            # Generate final report
            self.generate_tournament_report(results, team_stats, team_names)
            
            return {
                'results': results,
                'team_stats': team_stats,
                'teams': teams
            }
            
        except Exception as e:
            print(f"Error running tournament: {str(e)}")
            return None
    
    def generate_tournament_report(self, results, team_stats, team_names):
        """Generate a comprehensive tournament report"""
        try:
            print("\n" + "="*80)
            print("TOURNAMENT FINAL REPORT")
            print("="*80)
            
            # Sort teams by points, then by wins
            sorted_teams = sorted(team_stats.items(), 
                                key=lambda x: (x[1]['points'], x[1]['wins']), 
                                reverse=True)
            
            print("\nFINAL STANDINGS:")
            print("-" * 50)
            print(f"{'Rank':<5} {'Team':<25} {'Wins':<6} {'Losses':<8} {'Points':<8}")
            print("-" * 50)
            
            for rank, (team, stats) in enumerate(sorted_teams, 1):
                print(f"{rank:<5} {team:<25} {stats['wins']:<6} {stats['losses']:<8} {stats['points']:<8}")
            
            # Tournament champion
            champion = sorted_teams[0][0]
            print(f"\n🏆 TOURNAMENT CHAMPION: {champion}")
            print(f"   Wins: {sorted_teams[0][1]['wins']}")
            print(f"   Points: {sorted_teams[0][1]['points']}")
            
            # Statistics
            print(f"\nTOURNAMENT STATISTICS:")
            print("-" * 30)
            print(f"Total matches played: {len(results)}")
            print(f"Teams participated: {len(team_names)}")
            
            # High confidence matches
            high_confidence_matches = [r for r in results if r['confidence'] > 0.8]
            print(f"High confidence predictions (>80%): {len(high_confidence_matches)}")
            
            # Close matches
            close_matches = [r for r in results if abs(r['team1_win_probability'] - 0.5) < 0.1]
            print(f"Close matches (<60% probability): {len(close_matches)}")
            
            # Most dominant team (highest average win probability when they win)
            team_dominance = {}
            for result in results:
                winner = result['winner']
                if result['winner'] == result['team1']:
                    win_prob = result['team1_win_probability']
                else:
                    win_prob = result['team2_win_probability']
                
                if winner not in team_dominance:
                    team_dominance[winner] = []
                team_dominance[winner].append(win_prob)
            
            if team_dominance:
                avg_dominance = {team: np.mean(probs) for team, probs in team_dominance.items()}
                most_dominant = max(avg_dominance.items(), key=lambda x: x[1])
                print(f"Most dominant team: {most_dominant[0]} (avg win prob: {most_dominant[1]:.2%})")
            
            # Save detailed report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"tournament_report_{timestamp}.txt"
            
            try:
                with open(report_filename, 'w') as f:
                    f.write("CRICKET TOURNAMENT DETAILED REPORT\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write("FINAL STANDINGS:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{'Rank':<5} {'Team':<25} {'Wins':<6} {'Losses':<8} {'Points':<8}\n")
                    f.write("-" * 50 + "\n")
                    
                    for rank, (team, stats) in enumerate(sorted_teams, 1):
                        f.write(f"{rank:<5} {team:<25} {stats['wins']:<6} {stats['losses']:<8} {stats['points']:<8}\n")
                    
                    f.write(f"\nTOURNAMENT CHAMPION: {champion}\n")
                    f.write(f"Wins: {sorted_teams[0][1]['wins']}, Points: {sorted_teams[0][1]['points']}\n\n")
                    
                    f.write("DETAILED MATCH RESULTS:\n")
                    f.write("-" * 80 + "\n")
                    
                    for result in results:
                        f.write(f"Match {result['match_number']}: {result['team1']} vs {result['team2']}\n")
                        f.write(f"Winner: {result['winner']}\n")
                        f.write(f"Confidence: {result['confidence']:.2%}\n")
                        f.write(f"{result['team1']} win probability: {result['team1_win_probability']:.2%}\n")
                        f.write(f"{result['team2']} win probability: {result['team2_win_probability']:.2%}\n")
                        f.write(f"Team strengths - {result['team1']}: {result['team1_strength']:.2f}, {result['team2']}: {result['team2_strength']:.2f}\n")
                        f.write("-" * 40 + "\n")
                
                print(f"\nDetailed report saved to: {report_filename}")
            except Exception as e:
                print(f"Could not save report to file: {str(e)}")
            
            # Top 5 most exciting matches (closest probabilities)
            if results:
                exciting_matches = sorted(results, key=lambda x: abs(x['team1_win_probability'] - 0.5))[:5]
                
                print(f"\nTOP 5 MOST EXCITING MATCHES:")
                print("-" * 60)
                for i, match in enumerate(exciting_matches, 1):
                    prob_diff = abs(match['team1_win_probability'] - 0.5)
                    print(f"{i}. {match['team1']} vs {match['team2']}")
                    print(f"   Winner: {match['winner']} (Margin: {prob_diff:.1%})")
            
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"Error generating tournament report: {str(e)}")

def main():
    """Main function to run the enhanced cricket simulator"""
    try:
        print("Enhanced Cricket Match Simulator")
        print("="*50)
        
        # Get file paths from user
        print("\nDefault files:")
        print("- Player Statistics: 'IPL 2025 Player Statistics Clean.csv'")
        print("- Teams Data: Enter the path to your teams CSV file")
        
        statistics_file = input("\nEnter player statistics CSV path (or press Enter for default): ").strip()
        if not statistics_file:
            statistics_file = "IPL 2025 Player Statistics Clean.csv"
        
        teams_file = input("Enter teams CSV path: ").strip()
        
        if not teams_file:
            print("Teams CSV file is required!")
            return
        
        # Check if files exist
        if not os.path.exists(statistics_file):
            print(f"Error: Statistics file '{statistics_file}' not found!")
            create_sample = input("Would you like to create sample files for testing? (y/n): ").strip().lower()
            if create_sample in ['y', 'yes']:
                create_sample_files()
                statistics_file = "IPL 2025 Player Statistics.csv"
            else:
                return
        
        if not os.path.exists(teams_file):
            print(f"Error: Teams file '{teams_file}' not found!")
            if teams_file == "sample_teams.csv":
                create_sample = input("Would you like to create sample files for testing? (y/n): ").strip().lower()
                if create_sample in ['y', 'yes']:
                    create_sample_files()
                else:
                    return
            else:
                return
        
        # Create simulator and run tournament
        simulator = EnhancedCricketMatchSimulator()
        
        tournament_results = simulator.run_tournament(teams_file, statistics_file)
        
        if tournament_results:
            print("\nTournament completed successfully!")
            
            # Ask if user wants to see specific match details
            while True:
                try:
                    show_details = input("\nWould you like to see details of a specific match? (y/n): ").strip().lower()
                    if show_details in ['n', 'no']:
                        break
                    elif show_details in ['y', 'yes']:
                        match_num = int(input(f"Enter match number (1-{len(tournament_results['results'])}): "))
                        if 1 <= match_num <= len(tournament_results['results']):
                            match = tournament_results['results'][match_num - 1]
                            print(f"\nDETAILED MATCH ANALYSIS - Match {match_num}")
                            print("="*50)
                            print(f"Teams: {match['team1']} vs {match['team2']}")
                            print(f"Winner: {match['winner']}")
                            print(f"Confidence: {match['confidence']:.2%}")
                            print(f"Win Probabilities:")
                            print(f"  {match['team1']}: {match['team1_win_probability']:.2%}")
                            print(f"  {match['team2']}: {match['team2_win_probability']:.2%}")
                            print(f"Team Strengths:")
                            print(f"  {match['team1']}: {match['team1_strength']:.2f}")
                            print(f"  {match['team2']}: {match['team2_strength']:.2f}")
                        else:
                            print("Invalid match number!")
                except ValueError:
                    print("Please enter a valid number!")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
        else:
            print("Tournament failed to complete!")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error running main program: {str(e)}")

def create_sample_files():
    """Create sample files for demonstration"""
    try:
        print("Creating sample files...")
        
        # Sample player statistics
        sample_players = {
            'Player_Name': [
                'Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'Hardik Pandya', 'KL Rahul',
                'Jasprit Bumrah', 'Mohammed Shami', 'Ravindra Jadeja', 'Yuzvendra Chahal', 
                'Rishabh Pant', 'Shikhar Dhawan', 'Babar Azam', 'Fakhar Zaman', 
                'Mohammad Rizwan', 'Shadab Khan', 'Hasan Ali', 'Shaheen Afridi', 
                'Imad Wasim', 'Mohammad Hafeez', 'Sarfaraz Ahmed', 'Imam-ul-Haq', 
                'Haris Rauf', 'David Warner', 'Steve Smith', 'Pat Cummins',
                'Mitchell Starc', 'Glenn Maxwell', 'Marcus Stoinis', 'Alex Carey',
                'Adam Zampa', 'Josh Hazlewood', 'Kane Williamson', 'Trent Boult'
            ],
            'Matches_Batted': [254, 350, 243, 92, 42, 12, 8, 174, 5, 30, 167, 83, 50, 45, 52, 53, 7, 66, 218, 117, 11, 15, 145, 132, 45, 8, 98, 67, 89, 12, 5, 87, 23],
            'Runs_Scored': [12169, 10773, 9115, 1386, 1239, 19, 4, 2756, 0, 987, 6793, 3359, 1621, 1169, 458, 971, 7, 1152, 6614, 3045, 306, 34, 5455, 4378, 89, 15, 2236, 1543, 2567, 23, 8, 2101, 234],
            'Batting_Average': [59.07, 50.11, 48.63, 33.48, 51.25, 8.5, 2.0, 32.85, 0, 35.39, 45.12, 54.17, 36.02, 42.22, 15.26, 19.83, 3.5, 28.8, 32.44, 28.77, 34.0, 11.33, 45.3, 42.1, 23.4, 7.5, 32.8, 34.2, 31.8, 15.2, 4.0, 35.6, 28.9],
            'Batting_Strike_Rate': [93.17, 87.56, 88.90, 113.33, 86.21, 95.0, 40.0, 86.23, 0, 118.12, 96.27, 88.25, 85.73, 85.63, 73.91, 75.19, 70.0, 101.28, 86.65, 79.49, 79.27, 94.44, 142.3, 126.4, 145.2, 120.0, 154.8, 147.9, 123.4, 127.8, 80.0, 134.2, 156.7],
            'Centuries': [43, 10, 29, 0, 5, 0, 0, 1, 0, 0, 17, 13, 1, 1, 0, 0, 0, 0, 9, 4, 0, 0, 23, 18, 0, 0, 4, 2, 6, 0, 0, 8, 0],
            'Half_Centuries': [64, 73, 48, 5, 2, 0, 0, 16, 0, 2, 35, 16, 9, 7, 1, 3, 0, 7, 53, 18, 2, 0, 34, 29, 1, 0, 14, 8, 18, 0, 0, 12, 1],
            'Wickets_Taken': [4, 1, 8, 42, 0, 121, 180, 220, 121, 0, 0, 0, 0, 1, 89, 59, 89, 49, 54, 0, 0, 34, 2, 1, 167, 178, 45, 23, 1, 89, 134, 12, 156],
            'Bowling_Average': [49.25, 166.0, 61.37, 31.23, 0, 24.43, 25.06, 33.58, 25.09, 0, 0, 0, 0, 96.0, 33.89, 29.54, 25.65, 30.18, 35.07, 0, 0, 22.76, 67.5, 89.0, 22.5, 21.2, 28.9, 34.6, 78.0, 26.7, 23.4, 42.3, 24.8],
            'Economy_Rate': [6.11, 5.23, 7.27, 6.14, 0, 4.63, 5.28, 5.34, 4.85, 0, 0, 0, 0, 6.0, 5.53, 5.43, 5.74, 4.75, 5.24, 0, 0, 7.32, 8.2, 7.8, 5.1, 4.9, 7.8, 6.9, 8.5, 5.4, 4.7, 6.8, 5.2],
            'Catches_Taken': [131, 256, 78, 28, 15, 5, 3, 49, 17, 15, 72, 25, 16, 18, 20, 14, 2, 25, 64, 51, 3, 6, 67, 89, 12, 8, 34, 23, 45, 15, 3, 28, 8]
        }
        
        # Create player statistics CSV
        df_players = pd.DataFrame(sample_players)
        df_players.to_csv('IPL 2025 Player Statistics.csv', index=False)
        
        # Sample teams data
        teams_data = []
        teams = {
            'Mumbai Indians': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'Hardik Pandya', 'KL Rahul',
                              'Jasprit Bumrah', 'Mohammed Shami', 'Ravindra Jadeja', 'Yuzvendra Chahal', 
                              'Rishabh Pant', 'Shikhar Dhawan'],
            'Chennai Super Kings': ['Babar Azam', 'Fakhar Zaman', 'Mohammad Rizwan', 'Shadab Khan', 'Hasan Ali',
                                   'Shaheen Afridi', 'Imad Wasim', 'Mohammad Hafeez', 'Sarfaraz Ahmed', 
                                   'Imam-ul-Haq', 'Haris Rauf'],
            'Royal Challengers Bangalore': ['David Warner', 'Steve Smith', 'Pat Cummins', 'Mitchell Starc', 
                                           'Glenn Maxwell', 'Marcus Stoinis', 'Alex Carey', 'Adam Zampa', 
                                           'Josh Hazlewood', 'Kane Williamson', 'Trent Boult']
        }
        
        for team_name, players in teams.items():
            for player in players:
                teams_data.append({'Team': team_name, 'Player': player})
        
        df_teams = pd.DataFrame(teams_data)
        df_teams.to_csv('sample_teams.csv', index=False)
        
        print("Sample files created successfully:")
        print("- IPL 2025 Player Statistics.csv")
        print("- sample_teams.csv")
        
    except Exception as e:
        print(f"Error creating sample files: {str(e)}")

if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--create-samples":
            create_sample_files()
        else:
            # Check if required files exist, if not offer to create samples
            if not os.path.exists("IPL 2025 Player Statistics.csv"):
                create_samples = input("IPL 2025 Player Statistics.csv not found. Create sample files? (y/n): ").strip().lower()
                if create_samples in ['y', 'yes']:
                    create_sample_files()
                else:
                    print("Please ensure you have the required CSV files before running.")
                    sys.exit(1)
            
            main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")