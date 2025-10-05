/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xalan.internal.xsltc.compiler;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;

final class QName {

    public QName(String namespace, String prefix, String localname) {
    }

    public void clearNamespace();

    public String toString();

    public String getStringRep();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public String getLocalPart();

    public String getNamespace();

    public String getPrefix();

    public int hashCode();

    public String dump();
}
