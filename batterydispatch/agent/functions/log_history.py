import sqlite3
import pandas as pd
import numpy as np

def save_results(env, agent, history, reward, scenario=None, agent_name=None, notes=None):
    conn = sqlite3.connect('gym_battery_database.db')

    result = conn.execute('SELECT MAX(scenario_id) FROM grid_flow_output;')
    scenario_id = int(result.fetchone()[0]) + 1

    if scenario is None:
        scenario = input("Enter the scenario name (i.e. the load used): ")
    if agent is None:
        scenario = input("Enter the agent name, or y to accept {}: ".format(agent.name))
        if scenario.lower() == 'y':
            scenario = agent.name
    if notes is None:
        notes = input("Consider adding any notes: ")

    saved_time = pd.Timestamp.now()

    # Save the final grid_flow using entirely greedy policy
    DF = env.grid_flow.copy()

    DF['agent_state'] = [tuple(agent.discretize_space(np.array(s))) for s in DF.state]
    agent_state_hash_table = {hash(s): s for s in DF.agent_state}
    DF.agent_state = [hash(s) for s in DF.agent_state]

    state_hash_table = {hash(s): s for s in DF.state}
    DF.state = [hash(s) for s in DF.state]

    DF['reward'] = reward
    DF['agent'] = agent_name
    DF['scenario'] = scenario
    DF['episodes'] = len(history)
    DF['notes'] = notes
    DF['scenario_id'] = scenario_id
    DF['saved_timestamp'] = saved_time

    DF.to_sql('grid_flow_output', conn, if_exists='append')

    # Save the state-action value estimates
    val = agent.S_A_values.copy()
    val = pd.DataFrame.from_dict(val, orient='index')
    val = val.reset_index()
    val['state'] = [[i.level_0, i.level_1, i.level_2, i.level_3] for ix, i in val.iterrows()]
    val = val.rename(columns={"state": "agent_state"})
    val.index = val.agent_state
    val = val.drop(columns=['level_0', 'level_1', 'level_2', 'level_3', 'agent_state'])
    val.index = [tuple(x) for x in val.index]

    add_agent_state_hash = {hash(s): s for s in val.index if hash(s) not in agent_state_hash_table.keys()}
    agent_state_hash_table.update(add_agent_state_hash)

    val.index = [hash(s) for s in val.index]
    val['agent'] = agent_name
    val['scenario'] = scenario
    val['scenario_id'] = scenario_id
    val['saved_timestamp'] = saved_time
    val.to_sql('state_action_values', conn, if_exists='append')

    agent_state_hash_DF = pd.DataFrame.from_dict(agent_state_hash_table, orient='index',
                                                 columns=['hour', 'charge', 'load', 'demand'])
    agent_state_hash_DF['saved_timestamp'] = saved_time
    agent_state_hash_DF['state'] = agent_state_hash_DF.index
    try:
        agent_state_hash_DF = pd.read_sql('SELECT * FROM agent_states_hash;', conn).append(agent_state_hash_DF)
    except:
        print("Error reading in agent state hash table. Is this the first time you're running it?")
    agent_state_hash_DF.drop_duplicates(subset='state', inplace=True)
    agent_state_hash_DF.reset_index(drop=True, inplace=True)
    agent_state_hash_DF.saved_timestamp = pd.to_datetime(agent_state_hash_DF.saved_timestamp)
    #    conn.execute("DROP TABLE agent_states_hash;")
    try:
        agent_state_hash_DF.to_sql('agent_states_hash', conn, if_exists='replace', index=False)
    except:
        print("returning DF")
        return agent_state_hash_DF

    state_hash_DF = pd.DataFrame.from_dict(state_hash_table, orient='index',
                                           columns=['hour', 'charge', 'load', 'demand'])
    state_hash_DF['saved_timestamp'] = saved_time
    state_hash_DF['state'] = state_hash_DF.index
    try:
        state_hash_DF = pd.read_sql('SELECT * FROM states_hash;', conn).append(state_hash_DF)
    except:
        print("Error reading in state hash table. Is this the first time you're running it?")
    state_hash_DF.drop_duplicates(subset='state', inplace=True)
    state_hash_DF.reset_index(drop=True, inplace=True)
    state_hash_DF.saved_timestamp = pd.to_datetime(state_hash_DF.saved_timestamp)
    state_hash_DF.to_sql('states_hash', conn, if_exists='replace', index=False)

    # Save the history of performance by episode
    df_history = pd.DataFrame(history, columns=['episode_cnt', 'reward', 'new_demand', 'orig_reward', 'orig_demand'])
    df_history['saved_timestamp'] = saved_time
    df_history['agent'] = agent_name
    df_history['scenario'] = scenario
    df_history['scenario_id'] = scenario_id
    df_history['epsilon'] = agent.policy_args['eta']
    df_history['learning_rate'] = agent.learning_rate
    df_history.to_sql('history', conn, if_exists='append')

    conn.close()