const {test,expect}=require('@playwright/test');

test.beforeEach(async({page})=>{
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Interactive RAG Assistant'})).toBeVisible();
});

test('real Streamlit application exposes four workspace tabs',async({page})=>{
  await expect(page).toHaveTitle('DistributedRAG');
  for(const name of ['Chat','Documents','Retrieval Inspector','System']){
    await expect(page.getByRole('tab',{name:new RegExp(name)})).toBeVisible();
  }
  await page.getByRole('tab',{name:/System/}).click();
  await expect(page.getByText('DISCONNECTED',{exact:true})).toBeVisible();
});

test('document input validation is visible without calling a model',async({page})=>{
  await page.getByRole('tab',{name:/Documents/}).click();
  await page.getByRole('button',{name:/Ingest Document/}).click();
  await expect(page.getByText('Both Document ID and Content are required for ingestion.')).toBeVisible();
});

test('document ingestion, update and deletion complete through the UI',async({page})=>{
  await page.getByRole('tab',{name:/Documents/}).click();
  await page.getByRole('textbox',{name:'Document ID',exact:true}).fill('playwright-doc');
  await page.getByRole('textbox',{name:'Document Content (Markdown or Raw Text)',exact:true}).fill('Initial browser test document.');
  await page.getByRole('button',{name:/Ingest Document/}).click();
  await expect(page.getByText(/Added 1 chunks/)).toBeVisible();
  await page.getByRole('textbox',{name:'Document Content (Markdown or Raw Text)',exact:true}).fill('Updated browser test document.');
  await page.getByRole('button',{name:/Update Document/}).click();
  await expect(page.getByText(/updated successfully via Hirschberg/)).toBeVisible();
  await page.getByRole('textbox',{name:'Document ID to delete',exact:true}).fill('playwright-doc');
  await page.getByRole('button',{name:/Delete Document/}).click();
  await expect(page.getByText(/deleted successfully/)).toBeVisible();
});

test('backend rejection appears as an error instead of success',async({page})=>{
  await page.getByRole('tab',{name:/Documents/}).click();
  await page.getByRole('textbox',{name:'Document ID',exact:true}).fill('reject-fixture');
  await page.getByRole('textbox',{name:'Document Content (Markdown or Raw Text)',exact:true}).fill('Rejected synthetic document.');
  await page.getByRole('button',{name:/Ingest Document/}).click();
  await expect(page.getByText(/Synthetic ingestion rejection/)).toBeVisible();
});

test('chat renders the returned answer and source evidence',async({page})=>{
  await page.getByRole('textbox',{name:'Ask a question about the indexed corpus...'}).fill('How does the browser test work?');
  await page.getByRole('textbox',{name:'Ask a question about the indexed corpus...'}).press('Enter');
  await expect(page.getByText('Fixture answer: browser integration is working.',{exact:true})).toBeVisible();
  await page.getByText('📚 Sources (1)',{exact:true}).click();
  await expect(page.getByText('Synthetic source for frontend tests.',{exact:true})).toBeVisible();
});

test('cache controls display successful clearing',async({page})=>{
  await page.getByRole('tab',{name:/System/}).click();
  await page.getByRole('button',{name:/Clear Cache/}).click();
  await expect(page.getByText(/Cache cleared successfully/)).toBeVisible();
});
