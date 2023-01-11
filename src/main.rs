use rust_bert::{pipelines::{text_generation::{TextGenerationConfig, TextGenerationModel}, common::ModelType}};
use rust_bert::resources::RemoteResource;
use rust_bert::gpt_neo::GptNeoModelResources;
use rust_bert::gpt_neo::GptNeoMergesResources;
use rust_bert::gpt_neo::GptNeoVocabResources;
use rust_bert::gpt_neo::GptNeoConfigResources;

use stopwatch::Stopwatch;

fn main() {
    println!("Hello, world!");
    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_125M, // GPT_NEO_1_3B, GPT_NEO_2_7B
    ));
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_125M, // GPT_NEO_1_3B, GPT_NEO_2_7B
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_125M, // GPT_NEO_1_3B, GPT_NEO_2_7B
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_125M, // GPT_NEO_1_3B, GPT_NEO_2_7B
    ));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        num_beams: 5,
        no_repeat_ngram_size: 2,
        max_length: 200,
        // device: tch::Device::Cpu, // uncomment if you want to only use your CPU
        ..Default::default()
    };

    let model = TextGenerationModel::new(generate_config).unwrap();
    println!("Model successfully loaded.");

    loop {
        println!("Please enter `<prefix>/<text>`: ");

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();

        let split = line.split("/").collect::<Vec<&str>>();
        let slice = split.as_slice();

        let prefix = slice[0];
        let text = &slice[1..];

        let sw = Stopwatch::start_new();
        let output = model.generate(text, Some(prefix));
        println!("Model output: ");
        for sentence in output {
            println!("{}", sentence);
        }
        println!("Generating the output from the model took {}ms", sw.elapsed_ms());
    }
}
