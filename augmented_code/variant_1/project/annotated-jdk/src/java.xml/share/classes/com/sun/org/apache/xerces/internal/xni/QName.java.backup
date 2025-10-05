/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.xni;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;

public class QName implements Cloneable {

    public String prefix;

    public String localpart;

    public String rawname;

    public String uri;

    public QName() {
    }

    public QName(String prefix, String localpart, String rawname, String uri) {
    }

    public QName(QName qname) {
    }

    public void setValues(QName qname);

    public void setValues(String prefix, String localpart, String rawname, String uri);

    public void clear();

    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    public String toString();
}
