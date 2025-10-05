/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.keys.storage.implementations;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverSpi;

    @Positive
public class SingleCertificateResolver extends StorageResolverSpi {

    @Positive
    public SingleCertificateResolver(X509Certificate x509cert) {
    @Positive
    }

    @Positive
    public Iterator<Certificate> getIterator();

    @Positive
    static class InternalIterator implements Iterator<Certificate> {

    @Positive
        public InternalIterator(X509Certificate x509cert) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Certificate next();

    @Positive
        public void remove();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
