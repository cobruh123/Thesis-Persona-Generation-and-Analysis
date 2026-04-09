Genetic Algorithm 
python genetic_algorithm\ga_persona.py --base_url https://api.openai.com/v1 --api_key YOUR_OPENAI_API --model gpt-4o-mini --seed_personas genetic_algorithm\characters.json --harmful_prompts genetic_algorithm\train_jailbreak.json --out_dir runs/ga_gpt4omini --N 25 --M 3 --gens 10 --workers 3

Evaluation of personas (Getting target's responses -> Evaluating this responses by judge)
python .\persona_pep\run_evaluate.py --characters_dir DIRECTORY WITH CHARACTERS IN JSON --instructions_json rewritten_instructions.json --out persona_pep\target_responses.json
python persona_pep\judge.py --in_json target_responses.json --out_json persona_pep\judged.json --metrics_json metrics.json --resume 
