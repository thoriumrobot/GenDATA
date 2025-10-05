/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.impl.dtd;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;

public class XMLContentSpec {

    public static final short CONTENTSPECNODE_LEAF;

    public static final short CONTENTSPECNODE_ZERO_OR_ONE;

    public static final short CONTENTSPECNODE_ZERO_OR_MORE;

    public static final short CONTENTSPECNODE_ONE_OR_MORE;

    public static final short CONTENTSPECNODE_CHOICE;

    public static final short CONTENTSPECNODE_SEQ;

    public static final short CONTENTSPECNODE_ANY;

    public static final short CONTENTSPECNODE_ANY_OTHER;

    public static final short CONTENTSPECNODE_ANY_LOCAL;

    public static final short CONTENTSPECNODE_ANY_LAX;

    public static final short CONTENTSPECNODE_ANY_OTHER_LAX;

    public static final short CONTENTSPECNODE_ANY_LOCAL_LAX;

    public static final short CONTENTSPECNODE_ANY_SKIP;

    public static final short CONTENTSPECNODE_ANY_OTHER_SKIP;

    public static final short CONTENTSPECNODE_ANY_LOCAL_SKIP;

    public short type;

    public Object value;

    public Object otherValue;

    public XMLContentSpec() {
    }

    public XMLContentSpec(short type, Object value, Object otherValue) {
    }

    public XMLContentSpec(XMLContentSpec contentSpec) {
    }

    public XMLContentSpec(XMLContentSpec.Provider provider, int contentSpecIndex) {
    }

    public void clear();

    public void setValues(short type, Object value, Object otherValue);

    public void setValues(XMLContentSpec contentSpec);

    public void setValues(XMLContentSpec.Provider provider, int contentSpecIndex);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    public interface Provider {

        public boolean getContentSpec(int contentSpecIndex, XMLContentSpec contentSpec);
    }
}
