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
import java.io.ByteArrayInputStream;
    @Positive
import java.io.InputStream;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.security.Provider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import javax.xml.crypto.Data;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.NodeSetData;
    @Positive
import javax.xml.crypto.URIDereferencer;
    @Positive
import javax.xml.crypto.URIReferenceException;
    @Positive
import javax.xml.crypto.XMLCryptoContext;
    @Positive
import javax.xml.crypto.XMLStructure;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dom.DOMURIReference;
    @Positive
import javax.xml.crypto.dsig.Transform;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.keyinfo.RetrievalMethod;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public final class DOMRetrievalMethod extends DOMStructure implements RetrievalMethod, DOMURIReference {

    @Positive
    public DOMRetrievalMethod(String uri, String type, List<? extends Transform> transforms) {
    @Positive
    }

    @Positive
    public DOMRetrievalMethod(Element rmElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public String getURI();

    @Positive
    public String getType();

    @Positive
    public List<Transform> getTransforms();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    public Node getHere();

    @Positive
    public Data dereference(XMLCryptoContext context) throws URIReferenceException;

    @Positive
    public XMLStructure dereferenceAsXMLStructure(XMLCryptoContext context) throws URIReferenceException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
