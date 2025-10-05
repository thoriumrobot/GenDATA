/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.utils;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.Serializable;
    @Positive
import org.xml.sax.Attributes;
    @Positive
import org.xml.sax.helpers.AttributesImpl;

    @Positive
public class MutableAttrListImpl extends AttributesImpl implements Serializable {

    @Positive
    public MutableAttrListImpl() {
    @Positive
    }

    @Positive
    public MutableAttrListImpl(Attributes atts) {
    @Positive
    }

    @Positive
    public void addAttribute(String uri, String localName, String qName, String type, String value);

    @Positive
    public void addAttributes(Attributes atts);

    @Positive
    @Pure
    @Positive
    public boolean contains(String name);
    @Positive
}

// CFWR semantic augmentation - variant 0
