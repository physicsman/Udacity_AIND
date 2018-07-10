"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    
    

"""
Move Options:

'Move Options' attempts to return a more robust version of "Number of My Moves"
by returning the total number of spaces acessable up to 'depth' moves in the future.
this incresed the search time per call to the heuristic and over all only provides 
minor improvements with depth beyond 1 and diminishing returns after 3 levels
"""


"""
Center Score:

Returns the vertical + horzontal distance from the board center for the given player.
This is faster to compute than the square distance while still preserving the ordering
of scores computed. Since the the Minimax algorithim is only concerned with Min or Max
the ordering of scores is all that matters.

Generaly this is used to preferential keep the player near the center as trapping 
often occurs at the corners especially late in the game
"""


"""
My moves - Opp moves - Distance from center:

This custom score balances heuristic of maximizing the players moves and 
minimizing the opponents moves with the heuristic of avoiding the corners
of the board.
"""
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # No win situation
    if game.is_loser(player):
        return float("-inf")
    # Best outcome
    if game.is_winner(player):
        return float("inf")

    # Tuning parameter
    p0 = 0.5 
            
    # Improved Score
    own_len = float(len(game.get_legal_moves(player)))
    opp_len = float(len(game.get_legal_moves(game.get_opponent(player))))
    score = own_len - opp_len
    
    # Center Score
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    dist = float(abs(h - y) + abs(w - x))
    score -= p0*(dist)
    
    # My moves - Opp moves - Distance from board center
    return score

# Suggestion from reviewer
def custom_score_alt(game, player):

    # No win situation
    if game.is_loser(player):
        return float("-inf")
    # Best outcome
    if game.is_winner(player):
        return float("inf")

    # get current move count
    move_count = game.move_count

    # count number of moves available
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # calculate weight
    w = 10 / (move_count + 1)

    # return weighted delta of available moves
    return float(own_moves - (w * opp_moves))
    return score

"""
Find a way out:

This custom score simply favors more move options and near the end 
of the game it looks further and further out to find paths game states
with the most available outs
"""
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    # calculate weight
    d = game.move_count // 10
    
    # Moves avalible to Knight   
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    blanks = game.get_blank_spaces() # blank space on the board
    own_moves = set(game.get_legal_moves(player))
    own_moves -= set(game.get_legal_moves(game.get_opponent(player))) # My moves - Opp moves

    # Move Knight up to 'depth' moves away from initial position
    for _ in range(d):
        new_moves = []
        # Find all spaces avalible from current depth
        for r, c in own_moves:
            new_moves.extend([(r + dr, c + dc) for dr, dc in directions
                if (r + dr, c + dc) in blanks])
        #if not len(new_moves):
        #    break
        own_moves = set(new_moves)
    # Return a score that is total number of spaces that can be reached in 'depth' moves
    return float(len(own_moves))  


# Suggestion from reviewer
def custom_score_2_alt(game,player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = .0
    total_spaces = game.width * game.height
    remaining_spaces = len(game.get_blank_spaces())
    coefficient = float(total_spaces - remaining_spaces) / float(total_spaces)

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    for move in my_moves:
        isNearWall = 1 if (move[0] == 0 or move[0] == game.width - 1 or
            move[1] == 0 or move[1] == game.height - 1) else 0
        score += 1 - coefficient * isNearWall

    for move in opponent_moves:
        isNearWall = 1 if (move[0] == 0 or move[0] == game.width - 1 or
            move[1] == 0 or move[1] == game.height - 1) else 0
        score -= 1 - coefficient * isNearWall

    return score


"""
Fast - My moves
"""
def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
        
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
      
    return float(len(game.get_legal_moves(player))- game.move_count)


def inital():
    parameters = {}
    
    # custom_score
    parameters['p0_1'] = 0.5
    
    # custom_score_2
    #parameters['p0_2'] = 1
    
    # custom_score_3 59 59 57
    #parameters['p0_3'] = 0.5 
    
    return parameters

if __name__ == '__main__':
    parameters = inital()
    
class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.): # search_depth=3
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        # new parameters
        self.halt = False
        
        # testing parameters
        self.name = 'null'
        self.search_depths = []


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    #def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
    #    super(MinimaxPlayer, self).__init__( search_depth, score_fn, timeout)
    #    self.name = 'null'
        
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        if game.get_legal_moves(self):
            self.best_move = game.get_legal_moves(self)[0]
        else:
            return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            self.best_move = self.minimax(game, self.search_depth)
            self.search_depths.append(self.search_depth)
            return self.best_move

        except SearchTimeout: # Handle any actions required after timeout as needed
            pass
        
        # Return the best move from the last completed search iteration
        self.search_depths.append(self.search_depth)
        return self.best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        # The built in `max()` function can be used as argmax!
        if not game.get_legal_moves(self):
            return (-1, -1)
        best_move = max(game.get_legal_moves(self),
                   key=lambda m: self.min_value(game.forecast_move(m), depth - 1))
        return best_move
        
    
    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return self.score(game, self)
        v = float("inf")
        for move in moves:
            v = min(v, self.max_value(game.forecast_move(move), depth - 1))
        return v
        
    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return self.score(game, self)
        v = float("-inf")
        for move in moves:
            v = max(v, self.min_value(game.forecast_move(move), depth - 1))
        return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    #def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
    #    super(AlphaBetaPlayer, self).__init__( search_depth, score_fn, timeout)
    #    self.name = 'null'

    def get_move(self, game, time_left):
        # print("\nGame state:\n{}".format(game.to_string()))
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        if game.get_legal_moves(self):
            self.best_move = game.get_legal_moves(self)[0]
        else:
            return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            #for depth in range(self.search_depth):
            depth = 0
            self.halt = False
            while not self.halt:
                depth += 1
                self.best_move = self.alphabeta(game, depth)

        except SearchTimeout: # Handle any actions required after timeout as needed
            pass
        #self.search_depth = max(self.search_depth, depth)
        self.search_depths.append(depth)
        # Return the best move from the last completed search iteration
        return self.best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        if not game.get_legal_moves():
            return (-1, -1)
        chk = set()
        cmp = set([float("-inf"),float("inf")])
        for move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            chk.add(v)
            if v > alpha:
                alpha = v
                self.best_move = move

        if chk.issubset(cmp):
            self.halt = True

        return self.best_move
       
    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return self.score(game, self)
        v = float("inf")
        for move in moves:
            v = min(v, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v
        
    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return self.score(game, self)
        v = float("-inf")
        for move in moves:
            v = max(v, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
            
        return v