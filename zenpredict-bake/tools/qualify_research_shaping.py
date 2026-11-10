#!/usr/bin/env python3
"""Compare existing research shaping with consumer input programs, without labels.

The caller supplies the original canonical shaping.rs as an independent oracle.
Run under run-heavy. Output is create-new and includes exact source, input bakes,
programs, deterministic cases, migrated bakes, hashes, command and receipts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

RUST = r'''
use std::{error::Error,fs};
use zenpredict::{Model,Predictor};
use zenpredict::input_program::{InputOp,InputProgram,DamageShape,ReferenceGate};
use zenpredict_bake::input_program::attach_input_program;
#[path="research_wire.rs"] mod oracle;
enum Term { Direct(usize), Product(usize,usize), Shape(usize,Option<usize>,oracle::DamageShape), Separate(usize,usize,oracle::DamageShape,oracle::ReferenceGate) }
fn main()->Result<(),Box<dyn Error>> {
 let args:Vec<_>=std::env::args().collect();let out=std::path::Path::new(&args[1]);
 let bytes=fs::read(out.join("original.bin"))?;let model=Model::from_bytes(&bytes)?;
 let metadata=model.metadata();let text=metadata.get_utf8("zensim.derived_inputs")?;
 let mut lines=text.lines();let header=lines.next().ok_or("missing declaration")?;
 if header!="zensim-derived-inputs v1" && header!="zensim-derived-inputs research-shaping-v1" {return Err("unknown declaration schema".into());}
 let mut nodes=Vec::new();let mut outputs=Vec::new();let mut terms=Vec::new();
 // Source feature identity is intentionally supplied by the metric declaration.
 let mut sources=std::collections::BTreeMap::new();let mut source_count=0u32;
 for line in lines {
  let t:Vec<_>=line.split_whitespace().collect();if t.is_empty(){return Err("empty declaration line".into());}
  let count=match t[0] {"in"|"shape"=>1,"product"|"shape-product"|"separate"=>2,_=>return Err("unknown declaration op".into())};
  if t.len()<1+count{return Err("truncated declaration".into());}
  let mut ids=Vec::new();let mut refs=Vec::new();
  for value in &t[1..1+count] {
   let id=value.parse::<u32>()?;source_count=source_count.max(id.checked_add(1).ok_or("source overflow")?);
   let node=*sources.entry(id).or_insert_with(||{let n=nodes.len() as u32;nodes.push(InputOp::Source {index:id});n});
   ids.push(id as usize);refs.push(node);
  }
  let current=match t[0] {
   "in"=>{terms.push(Term::Direct(ids[0]));refs[0]},
   "product"=>{terms.push(Term::Product(ids[0],ids[1]));let n=nodes.len() as u32;nodes.push(InputOp::Product {left:refs[0],right:refs[1]});n},
   "shape"|"shape-product"|"separate"=>{
    let start=1+count;if t.len()<start+5{return Err("missing shape parameters".into());}
    let old=oracle::DamageShape::parse(&t[start..start+5])?;let p=old.parameters();let shape=DamageShape::new([p[0],p[1]],[p[2],p[3],p[4]])?;
    let input=if t[0]=="shape-product" {let n=nodes.len() as u32;nodes.push(InputOp::Product {left:refs[0],right:refs[1]});n} else {refs[0]};
    let n=nodes.len() as u32;nodes.push(InputOp::Shape {input,shape});
    if t[0]=="separate" {
     if t.len()!=start+8{return Err("invalid separate declaration".into());}
     let gate=oracle::ReferenceGate::parse(&t[start+5..start+8])?;let g=nodes.len() as u32;nodes.push(InputOp::Gate {input:refs[1],gate:ReferenceGate::new(gate.parameters())?});
     let product=nodes.len() as u32;nodes.push(InputOp::Product {left:n,right:g});terms.push(Term::Separate(ids[0],ids[1],old,gate));product
    } else {terms.push(Term::Shape(ids[0],if count==2 {Some(ids[1])} else {None},old));n}
   },_=>unreachable!()
  };outputs.push(current);
 }
 let program=InputProgram::new(source_count,nodes,outputs)?;
 let attached=attach_input_program(&bytes,&program)?;let migrated=Model::from_bytes(&attached)?;
 let mut old_predictor=Predictor::new(&model)?;let mut new_predictor=Predictor::new(&migrated)?;
 let mut scratch=vec![0.;program.scratch_len()];let mut compact=vec![0.;terms.len()];let mut cases=Vec::new();let mut state=0x12345678u32;
 for i in 0..64 {
  let source:Vec<f32>=(0..program.source_count()).map(|_|{state=state.wrapping_mul(1664525).wrapping_add(1013904223);match i {0=>0.,1=>1.,2=>0.5,_=>(state>>8) as f32/16777216.}}).collect();
  let prepared:Vec<f32>=terms.iter().map(|t|match *t {
   Term::Direct(d)=>source[d],Term::Product(d,r)=>source[d]*source[r],
   Term::Shape(d,r,s)=>s.eval(r.map_or(source[d],|r|source[d]*source[r])),
   Term::Separate(d,r,s,g)=>s.eval(source[d])*g.eval(source[r]),
  }).collect();
  program.apply(&source,&mut scratch,&mut compact)?;
  if compact.iter().map(|x|x.to_bits()).ne(prepared.iter().map(|x|x.to_bits())){return Err(format!("case {i}: input mismatch").into());}
  let a=old_predictor.predict_transformed(&prepared)?;let b=new_predictor.predict_from_source(&source)?;
  if a.iter().map(|x|x.to_bits()).ne(b.iter().map(|x|x.to_bits())){return Err(format!("case {i}: score mismatch").into());}
  cases.push(serde_json::json!({"source":source,"prepared":prepared,"predictions":b}));
 }
 fs::write(out.join("migrated.bin"),attached)?;fs::write(out.join("program.bin"),program.to_bytes())?;
 fs::write(out.join("cases.json"),serde_json::to_vec(&cases)?)?;
 fs::write(out.join("RESULT.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":"research-input-program-parity-v1","cases":cases.len(),"source_inputs":program.source_count(),"compact_inputs":model.n_inputs(),"nodes":program.nodes().len(),"prepared_inputs_bit_identical":true,"predictions_bit_identical":true,"dataset_labels_read":0}))?)?;
 println!("{}: {} cases; prepared inputs and scores bit-identical",out.display(),cases.len());Ok(())
}
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle-shaping-rs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('bakes',type=Path,nargs='+')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True,exist_ok=False)
    project=a.output/'oracle-project';(project/'src').mkdir(parents=True)
    (project/'src/main.rs').write_text(RUST)
    shutil.copy2(a.oracle_shaping_rs,project/'src/research_wire.rs')
    manifest='[package]\nname="research-input-parity"\nversion="0.0.0"\nedition="2024"\n[workspace]\n[dependencies]\n'
    for name in ('zenpredict','zenpredict-bake'):
        manifest+=name+' = {path='+json.dumps(str(repo/name))+'}\n'
    manifest+='serde_json="1.0.154"\n'
    (project/'Cargo.toml').write_text(manifest)
    command=['cargo','build','--offline','--manifest-path',str(project/'Cargo.toml')]
    subprocess.run(command,check=True)
    target=Path(os.environ.get('CARGO_TARGET_DIR',str(project/'target')))
    binary=target/'debug/research-input-parity'
    receipts=[]
    for index,bake in enumerate(a.bakes):
        output=a.output/('case-'+str(index));output.mkdir()
        shutil.copy2(bake,output/'original.bin')
        subprocess.run([str(binary),str(output)],check=True)
        receipts.append(dict(source=str(bake),result=json.loads((output/'RESULT.json').read_text())))
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file() and 'target' not in f.parts}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    names=subprocess.check_output(['git','ls-files','--','zenpredict','zenpredict-bake/src','zenpredict-bake/tools/qualify_research_shaping.py'],cwd=repo,text=True).splitlines()
    source_hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names if (repo/name).is_file()}
    (a.output/'QUALIFICATION.json').write_text(json.dumps(dict(command=command,repo_revision=revision,consumer_source_sha256=source_hashes,oracle_source=str(a.oracle_shaping_rs),receipts=receipts,sha256=hashes),indent=2)+'\n')
    print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
