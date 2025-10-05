/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.dtd;

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
import com.sun.org.apache.xerces.internal.util.XMLResourceIdentifierImpl;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLResourceIdentifier;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
    @Positive
import java.util.List;

    @Positive
public class XMLDTDDescription extends XMLResourceIdentifierImpl implements com.sun.org.apache.xerces.internal.xni.grammars.XMLDTDDescription {

    @Positive
    protected String fRootName;

    @Positive
    protected List<String> fPossibleRoots;

    @Positive
    public XMLDTDDescription(XMLResourceIdentifier id, String rootName) {
    @Positive
    }

    @Positive
    public XMLDTDDescription(String publicId, String literalId, String baseId, String expandedId, String rootName) {
    @Positive
    }

    @Positive
    public XMLDTDDescription(XMLInputSource source) {
    @Positive
    }

    @Positive
    public String getGrammarType();

    @Positive
    public String getRootName();

    @Positive
    public void setRootName(String rootName);

    @Positive
    public void setPossibleRoots(List<String> possibleRoots);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object desc);

    @Positive
    public int hashCode();
    @Positive
}
