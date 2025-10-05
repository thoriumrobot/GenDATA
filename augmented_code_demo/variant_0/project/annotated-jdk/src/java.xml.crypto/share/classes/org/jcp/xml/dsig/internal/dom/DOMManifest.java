/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package org.jcp.xml.dsig.internal.dom;

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
import javax.xml.crypto.*;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.*;
    @Positive
import java.security.Provider;
    @Positive
import java.util.*;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public final class DOMManifest extends DOMStructure implements Manifest {

    @Positive
    public DOMManifest(List<? extends Reference> references, String id) {
    @Positive
    }

    @Positive
    public DOMManifest(Element manElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public String getId();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static List<Reference> getManifestReferences(Manifest mf);

    @Positive
    @Override
    @Positive
    public List<Reference> getReferences();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
