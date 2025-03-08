from src import utils, llm, config
from torch.utils.data import DataLoader
import pandas as pd

def standard_prompting(test, tokenizer, model, prompt_template, batch_size=1, max_new_tokens=50):
    test_data = llm.MISSINFODataset(tokenizer=tokenizer, df=test, prompt_template=prompt_template)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    predicts, labels = llm.make_the_generation(model=model, tokenizer=tokenizer,
                                               data_loader=test_dataloader, max_new_tokens=max_new_tokens)
    processed_predicts = utils.output_processor(predicts)
    clf_report = utils.evaluation_report(y_true=labels, y_pred=processed_predicts)
    return predicts, processed_predicts, clf_report
    
    

fakehealth_train, fakehealth_test = utils.load_datasets(path="dataset", dataset_name="FakeHealth")
recovery_train, recovery_test = utils.load_datasets(path="dataset", dataset_name="ReCOVery")

fakehealth_all = pd.concat([fakehealth_train, fakehealth_test], ignore_index=True)
recovery_all = pd.concat([recovery_train, recovery_test], ignore_index=True)

llm_for_experimentations = [
    ['Qwen', llm.load_qwen_llm],
    ['Falcon', llm.load_falcon_llm], 
    ['Phi', llm.load_phi_llm],
    ['Llama3', llm.load_llama_llm],
]

for llms in llm_for_experimentations:
    prompt_template = config.standardized_prompt_template
    print(f"experimentation over:{llms[0]}")
    tokenizer, model, llm_path = llms[1]()
    
    ##################################################### Test based Evaluation!
    predicts, processed_predicts, clf_report = standard_prompting(test=recovery_test,
                                                               tokenizer=tokenizer,
                                                               model=model,
                                                               prompt_template=prompt_template)
    recovery_report_dict = {
         "llm": llm_path,
         "clf_report": clf_report,
         "generations": predicts,
         "processed_generations": processed_predicts,
    }
    utils.write_json(data=recovery_report_dict, path=f"results/{llms[0]}-ReCOVery-standard-prompting.json")
    predicts, processed_predicts, clf_report = standard_prompting(test=fakehealth_test,
                                                               tokenizer=tokenizer,
                                                               model=model,
                                                               prompt_template=prompt_template)
    fakehealth_report_dict = {
         "llm": llm_path,
         "clf_report": clf_report,
         "generations": predicts,
         "processed_generations": processed_predicts,
    }
    utils.write_json(data=fakehealth_report_dict, path=f"results/{llms[0]}-FakeHealth-standard-prompting.json")
    
    ##################################################### ALL dataset based Evaluation!
    predicts, processed_predicts, clf_report = standard_prompting(test=recovery_all,
                                                                  tokenizer=tokenizer,
                                                                  model=model,
                                                                  prompt_template=prompt_template)
    recovery_report_dict = {
        "llm": llm_path,
        "clf_report": clf_report,
        "generations": predicts,
        "processed_generations": processed_predicts,
    }
    utils.write_json(data=recovery_report_dict, path=f"results/{llms[0]}-ReCOVery-all-standard-prompting.json")
    predicts, processed_predicts, clf_report = standard_prompting(test=fakehealth_all,
                                                              tokenizer=tokenizer,
                                                              model=model,
                                                              prompt_template=prompt_template)
    fakehealth_report_dict = {
        "llm": llm_path,
        "clf_report": clf_report,
        "generations": predicts,
        "processed_generations": processed_predicts,
    }
    utils.write_json(data=fakehealth_report_dict, path=f"results/{llms[0]}-FakeHealth-all-standard-prompting.json")
    
