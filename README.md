# Reinforcement Learning Generator-Evaluator Architecture for Question Generation. 
  
We present in this work a fully Transformer-based Generator-Discriminator architecture for question generation. Transformer has been identified as a universal approximator of any sequence-to-sequence function. That is, a Transformer model which is deep enough can theoritically address the question generation task. For the sake of diversity --on the semantic sense-- at the inference, we experimented the Generator-discriminator architecture to address the question generation problem. T5 was used as the generator and the mixture of BLEU and the cosine similarity of the representation output by ELECTRA-discriminator was used as the discriminator.



## Description
### Model Architecture
![alt text](https://github.com/lkwate/neural-question-generation/blob/master/images/model-rl.png)

## Dependencies

```
torch
transformers
sentencepiece
pytorch_lightning
nltk
collections
bert-score

```

## Dataset
SQUAD v.2

## Evaluation
We used [NUBIA](https://wl-research.github.io/blog/2020/04/29/introducing-nubia.html) to evaluate our model, we got the following results
<table>
  <tr>
    <th>Model</th>
    <th> BLEU</th>
    <th> BERTScore </th>
    <th> Semantic relation </th>
    <th> Logical Agreement </th>
  </tr>
  <tr>
    <th>T5-RL</th>
    <th> 22.05</th>
    <th> 52.62 </th>
    <th> 61.31 </th>
    <th> 42.52 </th>
  </tr>
</table>



## Some Results

<table>
  <tr>
    <th>Context</th>
    <th> When Hitler launched World War II by invading Poland, British and French officials in Cameroon seized the German-owned plantations (1939-40). After the fall of France (June 1940), the country's colonial dependencies proclaimed loyalty to Marshal Petain' new Vichy regime. Central Africa was an exception</th>
    <th>answer</th>
  </tr>
  <tr>
    <td rowspan="6">Question predicted</td>
    <td>Who launched World War II by invading Poland?</td>
    <td>Hitler</td>
  </tr>
  <tr>
    <td>What war did Hitler begin by invading Poland?</td>
    <td>World War II</td>
  </tr>
  <tr>
    <td>What country did Hitler invade in World War II?</td>
    <td>Poland</td>
  </tr>
  <tr>
    <td>Who did Cameroon's colonial dependencies proclaim loyalty to?</td>
    <td>Marshal Petain</td>
  </tr>
  <tr>
    <td>What region was an exception to the Vichy regime?</td>
    <td>Central Africa</td>
  </tr>
  <tr>
    <td>What regime did the colonial dependencies of Cameroon resign to?</td>
    <td>Vichy</td>
  </tr>
</table>


<table>
  <tr>
    <th>Context</th>
    <th>The Douala-Yaoundé railway line, begun under the German regime, had been completed. Thousands of workers were forcibly deported to this site to work fifty-four hours a week. Workers also suffered from lack of food and the massive presence of mosquitoes. In 1925, the mortality rate on the site was 61.7%. However, the other sites were not as deadly, although working conditions were generally very harsh</th>
    <th>answer</th>
  </tr>
  <tr>
    <td rowspan="4">Question predicted</td>
    <td>What was the mortality rate in 1925?</td>
    <td>61.7%</td>
  </tr>
  <tr>
    <td>Which regime began the Douala-Yaoundé railway?</td>
    <td>German</td>
  </tr>
  <tr>
    <td>Which railroad line in the German era was completed in 1925?</td>
    <td>Douala</td>
  </tr>
  <tr>
    <td>How many workers were deported to this railroad site?</td>
    <td>Thousands</td>
  </tr>
</table>
