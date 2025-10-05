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
import java.io.OutputStream;
    @Positive
import java.security.InvalidAlgorithmParameterException;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.Provider;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import javax.xml.crypto.Data;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.XMLCryptoContext;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.Transform;
    @Positive
import javax.xml.crypto.dsig.TransformException;
    @Positive
import javax.xml.crypto.dsig.TransformService;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.dom.DOMSignContext;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public class DOMTransform extends DOMStructure implements Transform {

    @Positive
    protected TransformService spi;

    @Positive
    public DOMTransform(TransformService spi) {
    @Positive
    }

    @Positive
    public DOMTransform(Element transElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public final AlgorithmParameterSpec getParameterSpec();

    @Positive
    public final String getAlgorithm();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    public Data transform(Data data, XMLCryptoContext xc) throws TransformException;

    @Positive
    public Data transform(Data data, XMLCryptoContext xc, OutputStream os) throws TransformException;

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
    Data transform(Data data, XMLCryptoContext xc, DOMSignContext context) throws MarshalException, TransformException;
    @Positive
}

// CFWR semantic augmentation - variant 1
