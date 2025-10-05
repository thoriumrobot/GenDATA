/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.dom;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import org.w3c.dom.DOMStringList;

    @Positive
public class DOMStringListImpl implements DOMStringList {

    @Positive
    public DOMStringListImpl() {
    @Positive
    }

    @Positive
    public DOMStringListImpl(List<String> params) {
    @Positive
    }

    @Positive
    public String item(int index);

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(String param);

    @Positive
    public void add(String param);
    @Positive
}
