#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <string>
#include <random>
#include <iomanip>

// --- Configuration ---
const int GAMES_TO_PLAY = 10000;
const int SEARCH_DEPTH = 2;
const int BOARD_SIZE = 9;
const int MAX_TURNS = 200;

using namespace std;

// --- Structures ---

enum MoveType { MOVE_PAWN, MOVE_WALL };
enum Orientation { H, V };

struct Wall {
    int x, y;
    Orientation orientation;
    
    string toString() const {
        return (orientation == H ? "H" : "V") + to_string(x) + to_string(y);
    }
};

struct Player {
    int id;
    int x, y;
    int walls;
    int goal_y;
    string name;
};

struct Move {
    MoveType type;
    int x, y;
    Orientation orientation; // Only for walls
    
    // String format: "PAWN,x,y,0" or "WALL,x,y,H"
    string toCSV() const {
        if (type == MOVE_PAWN) return "PAWN," + to_string(x) + "," + to_string(y) + ",0";
        return "WALL," + to_string(x) + "," + to_string(y) + "," + (orientation == H ? "H" : "V");
    }
};

struct GameState {
    vector<Player> players;
    vector<Wall> walls;
    int turn; // 0 or 1
    int winner; // -1 if none

    GameState() {
        players.push_back({0, 4, 8, 10, 0, "P1 (Distance Only)"});
        players.push_back({1, 4, 0, 10, 8, "P2 (Wall Conserv)"});
        turn = 0;
        winner = -1;
    }
};

// Data structure for ML training
struct TrajectoryStep {
    int gameId;
    int turnNumber;
    int activePlayer;
    
    // State Features
    int p1_x, p1_y, p1_walls;
    int p2_x, p2_y, p2_walls;
    string wallString; // Compact string of all existing walls "H23;V44;..."
    
    // Label 1: Policy (The move actually taken)
    Move moveTaken;
    
    // Label 2: Value (The eventual winner of this game)
    int finalWinner; 
};

// --- Game Logic Class ---

class QuoridorSim {
private:
    std::mt19937 rng; 

public:
    QuoridorSim() {
        rng.seed(std::time(nullptr));
    }

    bool isValidCoord(int x, int y) const {
        return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
    }

    bool isPathBlocked(int x1, int y1, int x2, int y2, const vector<Wall>& walls) const {
        if (y1 == y2) { // Horizontal move
            int gap_x = min(x1, x2);
            int gap_y = y1;
            for (const auto& w : walls) {
                if (w.orientation == V && w.x == gap_x && (w.y == gap_y || w.y == gap_y - 1))
                    return true;
            }
        } else if (x1 == x2) { // Vertical move
            int gap_x = x1;
            int gap_y = min(y1, y2);
            for (const auto& w : walls) {
                if (w.orientation == H && w.y == gap_y && (w.x == gap_x || w.x == gap_x - 1))
                    return true;
            }
        }
        return false;
    }

    int bfsDistance(int start_x, int start_y, int target_y, const vector<Wall>& walls) const {
        if (start_y == target_y) return 0;
        deque<pair<int, int>> q;
        q.push_back({start_x, start_y});
        vector<vector<int>> dist(BOARD_SIZE, vector<int>(BOARD_SIZE, -1));
        dist[start_y][start_x] = 0;

        int dx[] = {0, 0, -1, 1};
        int dy[] = {-1, 1, 0, 0};

        while (!q.empty()) {
            pair<int, int> curr = q.front();
            q.pop_front();
            int cx = curr.first; int cy = curr.second;
            int d = dist[cy][cx];
            if (cy == target_y) return d;

            for (int i = 0; i < 4; i++) {
                int nx = cx + dx[i]; int ny = cy + dy[i];
                if (isValidCoord(nx, ny) && dist[ny][nx] == -1) {
                    if (!isPathBlocked(cx, cy, nx, ny, walls)) {
                        dist[ny][nx] = d + 1;
                        q.push_back({nx, ny});
                    }
                }
            }
        }
        return 9999; 
    }

    bool isValidWall(int x, int y, Orientation orientation, const vector<Wall>& currentWalls, const vector<Player>& players) const {
        if (x < 0 || x >= BOARD_SIZE - 1 || y < 0 || y >= BOARD_SIZE - 1) return false;
        for (const auto& w : currentWalls) {
            if (w.x == x && w.y == y) return false; 
            if (orientation == H) {
                if (w.orientation == H && w.y == y && abs(w.x - x) <= 1) return false;
            } else {
                if (w.orientation == V && w.x == x && abs(w.y - y) <= 1) return false;
            }
        }
        vector<Wall> tempWalls = currentWalls;
        tempWalls.push_back({x, y, orientation});
        if (bfsDistance(players[0].x, players[0].y, players[0].goal_y, tempWalls) >= 9999) return false;
        if (bfsDistance(players[1].x, players[1].y, players[1].goal_y, tempWalls) >= 9999) return false;
        return true;
    }

    vector<Move> getPossibleMoves(int playerIdx, const GameState& state) const {
        vector<Move> moves;
        const Player& p = state.players[playerIdx];
        const Player& opp = state.players[1 - playerIdx];
        int dx[] = {0, 0, -1, 1};
        int dy[] = {-1, 1, 0, 0};

        // 1. Pawn Moves
        for (int i = 0; i < 4; i++) {
            int nx = p.x + dx[i]; int ny = p.y + dy[i];
            if (isValidCoord(nx, ny)) {
                if (!isPathBlocked(p.x, p.y, nx, ny, state.walls)) {
                    if (nx == opp.x && ny == opp.y) {
                        int jump_x = nx + dx[i]; int jump_y = ny + dy[i];
                        if (isValidCoord(jump_x, jump_y) && !isPathBlocked(nx, ny, jump_x, jump_y, state.walls)) {
                            moves.push_back({MOVE_PAWN, jump_x, jump_y, H});
                        } else {
                            int ddx[2], ddy[2];
                            if (dx[i] == 0) { ddx[0] = -1; ddy[0] = 0; ddx[1] = 1; ddy[1] = 0; } 
                            else { ddx[0] = 0; ddy[0] = -1; ddx[1] = 0; ddy[1] = 1; }
                            for (int k = 0; k < 2; k++) {
                                int diag_x = nx + ddx[k]; int diag_y = ny + ddy[k];
                                if (isValidCoord(diag_x, diag_y) && !isPathBlocked(nx, ny, diag_x, diag_y, state.walls)) {
                                    moves.push_back({MOVE_PAWN, diag_x, diag_y, H});
                                }
                            }
                        }
                    } else {
                        moves.push_back({MOVE_PAWN, nx, ny, H});
                    }
                }
            }
        }

        // 2. Wall Moves
        if (p.walls > 0) {
            int r = 2;
            for (const auto& pl : state.players) {
                int min_x = max(0, pl.x - r); int max_x = min(BOARD_SIZE - 2, pl.x + r);
                int min_y = max(0, pl.y - r); int max_y = min(BOARD_SIZE - 2, pl.y + r);
                for (int y = min_y; y <= max_y; y++) {
                    for (int x = min_x; x <= max_x; x++) {
                        if (isValidWall(x, y, H, state.walls, state.players)) moves.push_back({MOVE_WALL, x, y, H});
                        if (isValidWall(x, y, V, state.walls, state.players)) moves.push_back({MOVE_WALL, x, y, V});
                    }
                }
            }
        }
        return moves;
    }

    double evaluate(const GameState& state, int rootPlayerIdx) const {
        int p1_dist = bfsDistance(state.players[0].x, state.players[0].y, state.players[0].goal_y, state.walls);
        int p2_dist = bfsDistance(state.players[1].x, state.players[1].y, state.players[1].goal_y, state.walls);
        if (rootPlayerIdx == 0) {
            // P1 Heuristic: Distance Only
            // Minimize P1 dist, Maximize P2 dist
            double score = (double)(p1_dist - p2_dist);
            score += (state.players[1].walls * 0.2) - (state.players[0].walls * 0.2);
            return (double)(p2_dist - p1_dist);
        } else {
            // P2 Heuristic: Distance + Wall Conservation
            double score = (double)(p1_dist - p2_dist);
            score += (state.players[1].walls * 0.2) - (state.players[0].walls * 0.2);
            return score;
        }
    }

    double minimaxRecursive(int depth, GameState state, double alpha, double beta, bool isMaximizing, int rootPlayerIdx) {
        if (depth == 0) return evaluate(state, rootPlayerIdx);
        if (state.players[0].y == state.players[0].goal_y) return rootPlayerIdx == 0 ? 1000.0 : -1000.0;
        if (state.players[1].y == state.players[1].goal_y) return rootPlayerIdx == 1 ? 1000.0 : -1000.0;

        int currentPlayer = isMaximizing ? rootPlayerIdx : (1 - rootPlayerIdx);
        vector<Move> moves = getPossibleMoves(currentPlayer, state);

        if (isMaximizing) {
            double maxEval = -99999.0;
            for (const auto& move : moves) {
                GameState nextState = state;
                if (move.type == MOVE_PAWN) {
                    nextState.players[currentPlayer].x = move.x;
                    nextState.players[currentPlayer].y = move.y;
                } else {
                    nextState.walls.push_back({move.x, move.y, move.orientation});
                    nextState.players[currentPlayer].walls--;
                }
                double evalVal = minimaxRecursive(depth - 1, nextState, alpha, beta, false, rootPlayerIdx);
                maxEval = max(maxEval, evalVal);
                alpha = max(alpha, evalVal);
                if (beta <= alpha) break;
            }
            return maxEval;
        } else {
            double minEval = 99999.0;
            for (const auto& move : moves) {
                GameState nextState = state;
                if (move.type == MOVE_PAWN) {
                    nextState.players[currentPlayer].x = move.x;
                    nextState.players[currentPlayer].y = move.y;
                } else {
                    nextState.walls.push_back({move.x, move.y, move.orientation});
                    nextState.players[currentPlayer].walls--;
                }
                double evalVal = minimaxRecursive(depth - 1, nextState, alpha, beta, true, rootPlayerIdx);
                minEval = min(minEval, evalVal);
                beta = min(beta, evalVal);
                if (beta <= alpha) break;
            }
            return minEval;
        }
    }

    Move runMinimax(GameState& state, int playerIdx) {
        double bestScore = -99999.0;
        Move bestMove = {MOVE_PAWN, -1, -1, H};
        bool foundMove = false;
        vector<Move> moves = getPossibleMoves(playerIdx, state);
        shuffle(moves.begin(), moves.end(), rng);

        for (const auto& move : moves) {
            GameState nextState = state;
            if (move.type == MOVE_PAWN) {
                nextState.players[playerIdx].x = move.x;
                nextState.players[playerIdx].y = move.y;
            } else {
                nextState.walls.push_back({move.x, move.y, move.orientation});
                nextState.players[playerIdx].walls--;
            }

            if (nextState.players[playerIdx].y == nextState.players[playerIdx].goal_y) return move;

            double score = minimaxRecursive(SEARCH_DEPTH - 1, nextState, -99999.0, 99999.0, false, playerIdx);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
                foundMove = true;
            }
        }
        if (!foundMove && !moves.empty()) return moves[0];
        return bestMove;
    }

    // --- Simulation Runner ---
    
    void runSimulation() {
        ofstream logFile("quoridor_results_cpp.txt");
        ofstream trajFile("quoridor_trajectories.csv");
        
        if (!trajFile.is_open()) { cerr << "Error opening trajectory file!" << endl; return; }

        // CSV Header for Trajectories
        trajFile << "GameID,Turn,ActivePlayer,P1_X,P1_Y,P1_Walls,P2_X,P2_Y,P2_Walls,Move_Type,Move_X,Move_Y,Move_Ori,Winner,Board_Walls\n";

        cout << "Starting C++ Simulation: " << GAMES_TO_PLAY << " Games" << endl;
        cout << left << setw(6) << "Game" << " | " << left << setw(25) << "Winner" << " | " << left << setw(6) << "Turns" << endl;

        int p1_wins = 0, p2_wins = 0, draws = 0;
        clock_t start_total = clock();

        for (int i = 1; i <= GAMES_TO_PLAY; ++i) {
            GameState state;
            int turns = 0;
            vector<TrajectoryStep> gameSteps;

            while (state.winner == -1 && turns < MAX_TURNS) {
                
                // 1. Record State Before Move
                TrajectoryStep step;
                step.gameId = i;
                step.turnNumber = turns;
                step.activePlayer = state.turn;
                step.p1_x = state.players[0].x; step.p1_y = state.players[0].y; step.p1_walls = state.players[0].walls;
                step.p2_x = state.players[1].x; step.p2_y = state.players[1].y; step.p2_walls = state.players[1].walls;
                
                string wStr = "";
                for(const auto& w : state.walls) wStr += w.toString() + ";";
                step.wallString = wStr;

                // 2. Decide Move
                Move move = runMinimax(state, state.turn);
                step.moveTaken = move;
                gameSteps.push_back(step);
                
                // 3. Apply Move
                if (move.type == MOVE_PAWN) {
                    state.players[state.turn].x = move.x;
                    state.players[state.turn].y = move.y;
                } else {
                    state.walls.push_back({move.x, move.y, move.orientation});
                    state.players[state.turn].walls--;
                }

                // 4. Check Win
                if (state.players[state.turn].y == state.players[state.turn].goal_y) {
                    state.winner = state.turn;
                } else {
                    state.turn = 1 - state.turn;
                }
                turns++;
            }

            // Stats Update
            string winnerName = (state.winner != -1) ? state.players[state.winner].name : "Draw";
            if (state.winner == 0) p1_wins++;
            else if (state.winner == 1) p2_wins++;
            else draws++;

            cout << left << setw(6) << i << " | " << left << setw(25) << winnerName << " | " << left << setw(6) << turns << endl;

            // Write Trajectories with Winner Label
            for (auto& step : gameSteps) {
                step.finalWinner = state.winner; // -1 for Draw, 0 for P1, 1 for P2
                
                trajFile << step.gameId << ","
                         << step.turnNumber << ","
                         << step.activePlayer << ","
                         << step.p1_x << "," << step.p1_y << "," << step.p1_walls << ","
                         << step.p2_x << "," << step.p2_y << "," << step.p2_walls << ","
                         << step.moveTaken.toCSV() << ","
                         << step.finalWinner << ","
                         << step.wallString << "\n";
            }
        }

        cout << "Sim Complete. Trajectories saved to 'quoridor_trajectories.csv'" << endl;
        logFile.close();
        trajFile.close();
    }
};

int main() {
    QuoridorSim sim;
    sim.runSimulation();
    return 0;
}