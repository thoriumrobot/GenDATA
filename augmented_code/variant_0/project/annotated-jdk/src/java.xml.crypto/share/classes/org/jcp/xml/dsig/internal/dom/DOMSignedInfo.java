/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @reserved * Positive comment block
    @DO * Positive NOT REMOVE OR ALTER!
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
import javax.xml.crypto.dsig.spec.RSAPSSParameterSpec;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.security.Provider;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import java.security.spec.MGF1ParameterSpec;
    @Positive
import java.security.spec.PSSParameterSpec;
    @Positive
import java.util.*;
    @Positive
import com.sun.org.apache.xml.internal.security.algorithms.implementations.SignatureBaseRSA;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;

    @Positive
public final class DOMSignedInfo extends DOMStructure implements SignedInfo {

    @Positive
    public DOMSignedInfo(CanonicalizationMethod cm, SignatureMethod sm, List<? extends Reference> references) {
    @Positive
    }

    @Positive
    public DOMSignedInfo(CanonicalizationMethod cm, SignatureMethod sm, List<? extends Reference> references, String id) {
    @Positive
    }

    @Positive
    public DOMSignedInfo(Element siElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public CanonicalizationMethod getCanonicalizationMethod();

    @Positive
    public SignatureMethod getSignatureMethod();

    @Positive
    public String getId();

    @Positive
    public List<Reference> getReferences();

    @Positive
    public InputStream getCanonicalizedData();

    @Positive
    public void canonicalize(XMLCryptoContext context, ByteArrayOutputStream bos) throws XMLSignatureException;

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
    @SuppressWarnings("unchecked")
    @Positive
    public static List<Reference> getSignedInfoReferences(SignedInfo si);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}
