from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/item", status_code=201)
def create_item(name: str, price: Union[float, None] = None):
    item = {"name": name}
    if price is not None:
        item["price"] = price
    return item

@app.put("/items/{item_id}", status_code=200)
def update_item(item_id: int, name: str, price: Union[float, None] = None):
    item = {"item_id": item_id, "name": name}
    if price is not None:
        item["price"] = price
    return item

@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    return {"message": f"Item {item_id} deleted"}


#I want a endpoint with 404 return
@app.post("/item-404", status_code=404)
def create_item_not_found(name: str):
    return {"message": "Item not found"}