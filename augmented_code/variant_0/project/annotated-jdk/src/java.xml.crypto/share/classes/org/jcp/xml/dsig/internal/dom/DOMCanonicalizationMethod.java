/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.security.Provider;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Set;
    @Positive
import org.w3c.dom.Element;
    @Positive
import javax.xml.crypto.*;
    @Positive
import javax.xml.crypto.dsig.*;

    @Positive
public class DOMCanonicalizationMethod extends DOMTransform implements CanonicalizationMethod {

    @Positive
    public DOMCanonicalizationMethod(TransformService spi) throws InvalidAlgorithmParameterException {
    @Positive
    }

    @Positive
    public DOMCanonicalizationMethod(Element cmElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public Data canonicalize(Data data, XMLCryptoContext xc) throws TransformException;

    @Positive
    public Data canonicalize(Data data, XMLCryptoContext xc, OutputStream os) throws TransformException;

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
