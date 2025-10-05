/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.xni;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;

    @Positive
public class QName implements Cloneable {

    @Positive
    public String prefix;

    @Positive
    public String localpart;

    @Positive
    public String rawname;

    @Positive
    public String uri;

    @Positive
    public QName() {
    @Positive
    }

    @Positive
    public QName(String prefix, String localpart, String rawname, String uri) {
    @Positive
    }

    @Positive
    public QName(QName qname) {
    @Positive
    }

    @Positive
    public void setValues(QName qname);

    @Positive
    public void setValues(String prefix, String localpart, String rawname, String uri);

    @Positive
    public void clear();

    @Positive
    public Object clone();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
