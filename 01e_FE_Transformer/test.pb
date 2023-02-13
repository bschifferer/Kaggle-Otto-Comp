feature {
  name: "session"
  type: INT
  int_domain {
    name: "session"
    min: 1
    max: 9249733 
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "aid_"
  value_count {
    min: 2
    max: 185
  }
  type: INT
  int_domain {
    name: "aid_"
    min: 1
    max: 1840500
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "type_"
  value_count {
    min: 2
    max: 185
  }
  type: INT
  int_domain {
    name: "type_"
    min: 1
    max: 4
    is_categorical: true
  }
  annotation {
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}