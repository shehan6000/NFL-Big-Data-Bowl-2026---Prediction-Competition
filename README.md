# NFL Big Data Bowl 2026 - Prediction Competition 


## Table of Contents
1. [Competition Overview](#competition-overview)
2. [Problem Statement](#problem-statement)
3. [Data Understanding](#data-understanding)
4. [Approach Strategy](#approach-strategy)
5. [Feature Engineering Guide](#feature-engineering-guide)
6. [Modeling Techniques](#modeling-techniques)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Implementation Pipeline](#implementation-pipeline)
9. [Advanced Techniques](#advanced-techniques)


---

## Competition Overview

### What You're Predicting
You need to predict the future positions (x, y coordinates) of NFL players during the time when the ball is in the air after a pass is thrown. This involves forecasting player movement across multiple frames (10 frames per second) until the ball lands or the pass is ruled incomplete.

### Key Challenge
The difficulty lies in predicting multi-agent dynamics where:
- **Offensive players** (especially the targeted receiver) move toward the ball landing location
- **Defensive players** track receivers and try to intercept the ball
- **Other offensive players** continue their routes or adjust based on the play
- **All players** react to each other's movements in real-time

### Competition Format
- **Training Phase**: Build models using historical 2023 NFL tracking data
- **Forecasting Phase**: Your model runs on live NFL games from Weeks 14-18 of the 2025 season
- **Live Leaderboard**: Updated weekly as games are played

---

## Problem Statement

### Given Information (At Prediction Time)
1. **Pre-throw tracking data**: All player positions, velocities, and orientations before the ball is thrown
2. **Targeted receiver**: Which offensive player is the intended target
3. **Ball landing location**: Where the ball will land (x, y coordinates)
4. **Number of frames**: How many future positions to predict (varies by play)

### What to Predict
For each player marked with `player_to_predict = True`:
- X-coordinate at each future frame
- Y-coordinate at each future frame
- Predictions needed for 1 to ~40 frames (0.1 to 4 seconds)

### Constraints
- Quick passes (< 0.5 seconds) are excluded
- Deflected passes and throwaways are excluded
- Field boundaries: X ∈ [0, 120] yards, Y ∈ [0, 53.3] yards

---

## Data Understanding

### Input Data Structure

#### Player Information
- **`nfl_id`**: Unique player identifier
- **`player_position`**: WR, CB, QB, etc.
- **`player_role`**: Targeted Receiver, Defensive Coverage, Passer, Other Route Runner
- **`player_side`**: Offense or Defense
- **`player_height`**: Format "ft-in" (e.g., "6-2")
- **`player_weight`**: Pounds
- **`player_birth_date`**: For calculating age

#### Tracking Data (Per Frame)
- **`x`**: Position along long axis (0-120 yards)
- **`y`**: Position along short axis (0-53.3 yards)
- **`s`**: Speed in yards/second
- **`a`**: Acceleration in yards/second²
- **`o`**: Player orientation (degrees)
- **`dir`**: Direction of movement (degrees)

#### Play Context
- **`game_id`**: Unique game identifier
- **`play_id`**: Play identifier (not unique across games)
- **`frame_id`**: Frame number (starts at 1 for each play)
- **`play_direction`**: "left" or "right"
- **`absolute_yardline_number`**: Distance from end zone
- **`ball_land_x`**, **`ball_land_y`**: Where the ball lands
- **`num_frames_output`**: How many frames to predict

### Output Data Structure
Simple format with actual positions:
- **`game_id`**, **`play_id`**, **`nfl_id`**, **`frame_id`**
- **`x`**, **`y`**: Ground truth positions (targets to predict)

### Submission Format
- **`id`**: Concatenated as `{game_id}_{play_id}_{nfl_id}_{frame_id}`
- **`x`**, **`y`**: Your predicted positions

---

## Approach Strategy

### Level 1: Simple Baseline (Start Here)
**Physics-Based Linear Extrapolation**

This assumes players continue moving in their current direction at their current speed.

**Rationale:**
- Players have momentum
- Direction and speed at throw time are informative
- Simple to implement and understand
- Provides a reasonable lower bound

**Limitations:**
- Ignores ball landing location
- No player interactions
- Doesn't account for role differences

### Level 2: Enhanced Physics Model
**Add Context-Aware Adjustments**

**Key Improvements:**
1. **Attraction to ball landing**: Players adjust trajectory toward where ball will land
2. **Role-based behavior**: Targeted receivers have stronger attraction
3. **Deceleration modeling**: Players slow down as they approach target
4. **Field constraints**: Ensure predictions stay within field boundaries

**Tunable Parameters:**
- Attraction strength (how much players move toward ball)
- Deceleration rate (how quickly players slow down)
- Role-specific multipliers (different behavior for different roles)

### Level 3: Machine Learning Approach
**Learn Patterns from Historical Data**

**Why ML Helps:**
- Captures non-linear relationships
- Learns role-specific behavior automatically
- Can incorporate complex interactions
- Handles edge cases better

**Model Options:**
- **Gradient Boosting**: Good for tabular features, fast training
- **Random Forests**: Robust, less prone to overfitting
- **Neural Networks**: Can capture complex patterns but needs more data

### Level 4: Advanced Sequence Models
**Temporal Pattern Recognition**

**Best for:**
- Learning trajectory curves
- Capturing momentum changes
- Understanding play evolution
- Multi-step prediction

**Architectures:**
- **LSTM/GRU**: Classic sequence models
- **Transformers**: Can handle variable-length sequences
- **Graph Neural Networks**: Model player interactions explicitly

---

## Feature Engineering Guide

### Essential Features

#### 1. Distance-Based Features
**Distance to Ball Landing**
- Most important feature
- Different meanings for different roles:
  - Targeted receiver: How far they need to go
  - Defense: Interception opportunity
  - Other offense: Usually less relevant

**Distance to Other Players**
- Nearest defender (for offensive players)
- Nearest receiver (for defensive players)
- Separation from coverage

#### 2. Directional Features
**Angle to Ball Landing**
- Calculate: `arctan2(ball_y - player_y, ball_x - player_x)`
- Shows if player is already moving toward ball

**Direction Difference**
- Difference between current direction and angle to ball
- Small difference = already moving right way
- Large difference = needs to change direction

**Velocity Toward Ball**
- Component of velocity in direction of ball
- Positive = moving toward, Negative = moving away

#### 3. Velocity Features
**Velocity Components**
- `vx = speed * cos(direction)`
- `vy = speed * sin(direction)`
- Useful for linear extrapolation

**Speed Categories**
- Stationary (< 1 yard/sec)
- Jogging (1-4 yards/sec)
- Running (4-7 yards/sec)
- Sprinting (> 7 yards/sec)

#### 4. Role-Based Features
**Binary Indicators**
- `is_targeted_receiver`: Most important offensive player
- `is_defensive_coverage`: Primary defenders
- `is_passer`: Usually stationary or moving slowly
- `is_other_route_runner`: Less predictable

**Role Interactions**
- Distance from targeted receiver (for defenders)
- Number of nearby defenders (for receiver)

#### 5. Temporal Features
**Time to Ball Landing**
- Can be approximated: `num_frames_output * 0.1` seconds
- Longer air time = more uncertainty

**Frame Position**
- Earlier frames: More influenced by momentum
- Later frames: More influenced by ball location

#### 6. Play Context Features
**Field Position**
- `absolute_yardline_number`: May affect player behavior
- Distance to sidelines: Constrains movement
- Red zone indicator: High-stakes area

**Play Direction**
- Standardize all plays to same direction
- Or encode as binary feature

#### 7. Player Physical Features
**Size and Age**
- Height (convert to inches)
- Weight
- Age (calculate from birth date)
- Speed potential varies by position and age

**Position Encoding**
- One-hot encode or use embeddings
- WR, CB, S have different movement patterns than linemen

### Advanced Feature Ideas

#### Relative Positioning
- Where player is relative to formation
- Distance from line of scrimmage
- Spread of offensive/defensive formations

#### Historical Player Patterns
- Average speed for this player/position
- Typical acceleration patterns
- Route tendencies

#### Game Situation Features
- Score differential (if available)
- Quarter/time remaining
- Down and distance

---

## Modeling Techniques

### Technique 1: Physics-Based Models

#### Pure Kinematics
**Equations:**
- Position: `x(t) = x₀ + v₀·t + ½·a·t²`
- For constant velocity: `x(t) = x₀ + v·t`

**Advantages:**
- No training required
- Interpretable
- Fast predictions
- Reasonable baseline

**Disadvantages:**
- Ignores context
- Linear assumptions
- No learning from data

#### Physics with Corrections
**Add Goal-Directed Movement:**
- Calculate attraction vector to ball
- Blend physics prediction with attraction
- Strength increases over time

**Pseudocode Logic:**
```
1. Calculate base position using current velocity
2. Calculate vector toward ball landing
3. Apply role-based attraction weight
4. Blend: final_pos = physics_pos + attraction_weight * ball_vector
5. Apply constraints (field boundaries, max speed)
```

### Technique 2: Regression Models

#### Predict Displacement Directly
**Target Variable:**
- Don't predict absolute position
- Predict displacement from current position
- Easier to learn: `Δx = x_future - x_current`

**Model Training:**
- Features: All engineered features
- Target: Displacement per frame or total displacement
- Separate models for x and y (or joint model)

#### Frame-by-Frame Prediction
**Approach:**
- For each future frame, predict displacement
- Accumulate displacements: `x_frame_n = x_0 + Σ(displacements)`
- Include frame number as feature

**Challenge:**
- Error accumulation over frames
- Later frames less accurate

#### Direct Multi-Output
**Approach:**
- Predict all frames simultaneously
- Output dimension = 2 × num_frames
- Learns correlations between frames

**Challenge:**
- Variable output size requires padding or separate models

### Technique 3: Sequence Models

#### LSTM/GRU Architecture
**Input Sequence:**
- All pre-throw frames for each player
- Features: [x, y, speed, direction, acceleration, ...]
- Sequence length varies by play

**Output:**
- Future positions for all frames after throw
- Can be autoregressive or direct

**Architecture Considerations:**
- Bidirectional for encoding input
- Many-to-many for sequence output
- Attention mechanisms for focusing on relevant frames

#### Encoder-Decoder Pattern
**Encoder:**
- Process input sequence (pre-throw tracking)
- Compress to fixed representation
- Include static context (role, ball location)

**Decoder:**
- Generate output sequence (future positions)
- Condition on encoded representation
- Can use teacher forcing during training

### Technique 4: Graph Neural Networks

#### Model Player Interactions
**Graph Structure:**
- Nodes: Players
- Edges: Proximity or role-based connections
- Features: Position, velocity, role

**Message Passing:**
- Each player "communicates" with nearby players
- Aggregate information from neighbors
- Update node states

**Advantages:**
- Explicitly models interactions
- Permutation invariant
- Captures team dynamics

**Implementation:**
- Libraries: PyTorch Geometric, DGL
- Can combine with temporal models
- Computational cost higher

### Technique 5: Ensemble Methods

#### Combine Multiple Approaches
**Why Ensemble:**
- Different models capture different patterns
- Reduces overfitting
- More robust predictions

**Simple Ensemble:**
- Train physics, ML, and sequence models
- Average predictions with weights
- Optimize weights on validation set

**Stacking:**
- Use model predictions as features
- Train meta-model to combine
- Can learn which model to trust when

---

## Evaluation Metrics

### Primary Metric: RMSE
**Root Mean Squared Error**

**Formula:**
```
RMSE = sqrt(mean((x_pred - x_true)² + (y_pred - y_true)²))
```

**Interpretation:**
- Average Euclidean distance error in yards
- RMSE of 1.0 = average 1 yard off
- Lower is better

**Characteristics:**
- Penalizes large errors more than small ones
- Sensitive to outliers
- Aggregated across all players and frames

### What Affects RMSE

#### Time Horizon
- Early frames: Easier to predict, smaller errors
- Later frames: More uncertainty, larger errors
- Long passes have higher error

#### Player Role
- Targeted receiver: Most predictable (moving to ball)
- Defensive coverage: Moderate predictability
- Other players: More variable

#### Play Characteristics
- Short quick passes: Lower error
- Deep throws: Higher error
- Congested areas: More interactions, higher error

### Validation Strategy

#### Time-Based Split
**Do this:**
- Train on earlier weeks
- Validate on later weeks
- Mimics test conditions (forecasting future)

**Don't do this:**
- Random split across all data
- Leads to data leakage

#### Cross-Validation Considerations
- Split by game or week, not by play
- Ensure no play appears in both train and validation
- Consider player-based splits for robustness

#### Metrics to Track
- Overall RMSE
- RMSE by frame number (early vs late)
- RMSE by player role
- RMSE by pass distance
- Percentage of predictions within 1 yard, 2 yards, etc.

---

## Implementation Pipeline

### Step 1: Data Preparation
1. Load all training week files
2. Combine into single dataframe
3. Handle missing values (rare but check)
4. Parse date/height formats
5. Create train/validation split by week

### Step 2: Exploratory Analysis
1. Visualize sample plays
2. Analyze distribution of:
   - Pass distances
   - Number of frames
   - Player speeds and accelerations
3. Examine role-based patterns
4. Check for data quality issues

### Step 3: Feature Engineering
1. Create all engineered features
2. Get last frame before throw for each player
3. Merge with static features
4. Normalize/scale if needed for ML
5. Save feature engineering pipeline

### Step 4: Model Development
1. **Start simple**: Implement physics baseline
2. **Evaluate**: Calculate RMSE on validation set
3. **Iterate**: Add improvements one at a time
4. **Compare**: Track which changes help
5. **Tune**: Optimize hyperparameters

### Step 5: Prediction Generation
1. Load test data
2. Apply same feature engineering
3. Generate predictions for all frames
4. Format correctly (game_play_player_frame)
5. Ensure all required IDs are present

### Step 6: Submission
1. Create submission CSV
2. Validate format matches sample
3. Check for missing predictions
4. Submit to leaderboard
5. Monitor live results during forecasting phase

---

## Advanced Techniques

### Handling Variable-Length Sequences

#### Problem
- Different plays have different number of:
  - Input frames (pre-throw tracking)
  - Output frames (time ball is in air)

#### Solutions
1. **Padding**: Pad to maximum length, use masking
2. **Bucketing**: Group similar lengths, separate models
3. **Per-frame models**: Predict each frame independently
4. **Attention**: Let model focus on relevant parts

### Incorporating Uncertainty

#### Probabilistic Predictions
- Predict distribution instead of point estimate
- Useful for understanding confidence
- Can use quantile regression or Bayesian methods

#### Ensemble for Uncertainty
- Variation across models indicates uncertainty
- High agreement = confident
- High disagreement = uncertain region

### Transfer Learning

#### Pre-training Ideas
- Train on all players, fine-tune by role
- Pre-train on 2023, fine-tune on 2024 data
- Learn general trajectory patterns first

#### Few-Shot Learning
- Some plays/situations rare in training
- Meta-learning approaches could help
- Learn to adapt quickly to new patterns

### Real-Time Considerations

#### Optimization
- Model must run fast for live forecasting
- Simple models advantageous
- Consider model complexity vs. accuracy tradeoff

#### Robustness
- Handle missing or noisy input data
- Graceful degradation if features unavailable
- Sanity checks on outputs

---

