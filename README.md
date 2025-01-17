# Robotics RL project ITHS

## Run the code
To run all of the code, including training model with PPO, fine_tuning, and testing the fine tuned model:

> [!NOTE]
> The main script takes about <5min to run on a AMD Ryzen 7 CPU.

Install the necessary packages (preferrably in a venv)


```bash
$ pip install -r requirements.txt
$ cd <root directory of project>
$ python ./main.py

```

### Test model
If you just want to test a fine tuned model, it is possible to only run the script "run_fine_tuned_model.py".  
Just be sure to have a model in the correct directory.

```bash
$ python ./run_model/run_fine_tuned_model.py
```

### Authors
*[Rasmus Berghäll](https://github.com/Crudeerz), [Joakim Roberg](https://github.com/Robergjo), [Max Färnström](https://github.com/arthead-git-user)*
