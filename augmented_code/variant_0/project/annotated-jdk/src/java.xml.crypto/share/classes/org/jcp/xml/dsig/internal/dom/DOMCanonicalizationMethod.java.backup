/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.OutputStream;
import java.security.InvalidAlgorithmParameterException;
import java.security.Provider;
import java.security.spec.AlgorithmParameterSpec;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.w3c.dom.Element;
import javax.xml.crypto.*;
import javax.xml.crypto.dsig.*;

public class DOMCanonicalizationMethod extends DOMTransform implements CanonicalizationMethod {

    public DOMCanonicalizationMethod(TransformService spi) throws InvalidAlgorithmParameterException {
    }

    public DOMCanonicalizationMethod(Element cmElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public Data canonicalize(Data data, XMLCryptoContext xc) throws TransformException;

    public Data canonicalize(Data data, XMLCryptoContext xc, OutputStream os) throws TransformException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();
}
