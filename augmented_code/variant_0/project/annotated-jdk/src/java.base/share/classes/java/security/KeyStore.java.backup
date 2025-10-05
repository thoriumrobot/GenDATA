/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.security;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.security.cert.CertificateException;
import java.security.spec.AlgorithmParameterSpec;
import java.util.*;
import javax.crypto.SecretKey;
import javax.security.auth.DestroyFailedException;
import javax.security.auth.callback.*;
import sun.security.util.Debug;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class KeyStore {

    public static interface LoadStoreParameter {

        @Nullable
        public ProtectionParameter getProtectionParameter();
    }

    public static interface ProtectionParameter {
    }

    public static class PasswordProtection implements ProtectionParameter, javax.security.auth.Destroyable {

        public PasswordProtection(char @Nullable [] password) {
        }

        public PasswordProtection(char @Nullable [] password, String protectionAlgorithm, @Nullable AlgorithmParameterSpec protectionParameters) {
        }

        @Nullable
        public String getProtectionAlgorithm();

        @Nullable
        public AlgorithmParameterSpec getProtectionParameters();

        public synchronized char @Nullable [] getPassword();

        public synchronized void destroy() throws DestroyFailedException;

        public synchronized boolean isDestroyed();
    }

    public static class CallbackHandlerProtection implements ProtectionParameter {

        public CallbackHandlerProtection(CallbackHandler handler) {
        }

        public CallbackHandler getCallbackHandler();
    }

    public static interface Entry {

        public default Set<Attribute> getAttributes();

        public interface Attribute {

            public String getName();

            public String getValue();
        }
    }

    public static final class PrivateKeyEntry implements Entry {

        public PrivateKeyEntry(PrivateKey privateKey, Certificate[] chain) {
        }

        public PrivateKeyEntry(PrivateKey privateKey, Certificate[] chain, Set<Attribute> attributes) {
        }

        public PrivateKey getPrivateKey();

        public Certificate[] getCertificateChain();

        public Certificate getCertificate();

        @Override
        public Set<Attribute> getAttributes();

        public String toString();
    }

    public static final class SecretKeyEntry implements Entry {

        public SecretKeyEntry(SecretKey secretKey) {
        }

        public SecretKeyEntry(SecretKey secretKey, Set<Attribute> attributes) {
        }

        public SecretKey getSecretKey();

        @Override
        public Set<Attribute> getAttributes();

        public String toString();
    }

    public static final class TrustedCertificateEntry implements Entry {

        public TrustedCertificateEntry(Certificate trustedCert) {
        }

        public TrustedCertificateEntry(Certificate trustedCert, Set<Attribute> attributes) {
        }

        public Certificate getTrustedCertificate();

        @Override
        public Set<Attribute> getAttributes();

        public String toString();
    }

    protected KeyStore(KeyStoreSpi keyStoreSpi, Provider provider, String type) {
    }

    public static KeyStore getInstance(String type) throws KeyStoreException;

    public static KeyStore getInstance(String type, String provider) throws KeyStoreException, NoSuchProviderException;

    public static KeyStore getInstance(String type, Provider provider) throws KeyStoreException;

    public static final String getDefaultType();

    public final Provider getProvider();

    public final String getType();

    @Nullable
    public final Key getKey(String alias, char[] password) throws KeyStoreException, NoSuchAlgorithmException, UnrecoverableKeyException;

    public final Certificate @Nullable [] getCertificateChain(String alias) throws KeyStoreException;

    @Nullable
    public final Certificate getCertificate(String alias) throws KeyStoreException;

    @Nullable
    public final Date getCreationDate(String alias) throws KeyStoreException;

    public final void setKeyEntry(String alias, Key key, char[] password, Certificate[] chain) throws KeyStoreException;

    public final void setKeyEntry(String alias, byte[] key, Certificate[] chain) throws KeyStoreException;

    public final void setCertificateEntry(String alias, Certificate cert) throws KeyStoreException;

    public final void deleteEntry(String alias) throws KeyStoreException;

    public final Enumeration<String> aliases() throws KeyStoreException;

    @Pure
    public final boolean containsAlias(String alias) throws KeyStoreException;

    public final int size() throws KeyStoreException;

    public final boolean isKeyEntry(String alias) throws KeyStoreException;

    public final boolean isCertificateEntry(String alias) throws KeyStoreException;

    @Nullable
    public final String getCertificateAlias(Certificate cert) throws KeyStoreException;

    public final void store(OutputStream stream, char[] password) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    public final void store(@Nullable LoadStoreParameter param) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    public final void load(@Nullable InputStream stream, char @Nullable [] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    public final void load(@Nullable LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Nullable
    public final Entry getEntry(String alias, @Nullable ProtectionParameter protParam) throws NoSuchAlgorithmException, UnrecoverableEntryException, KeyStoreException;

    public final void setEntry(String alias, Entry entry, @Nullable ProtectionParameter protParam) throws KeyStoreException;

    public final boolean entryInstanceOf(String alias, Class<? extends KeyStore.Entry> entryClass) throws KeyStoreException;

    public static final KeyStore getInstance(File file, char @Nullable [] password) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    public static final KeyStore getInstance(File file, @Nullable LoadStoreParameter param) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    public abstract static class Builder {

        protected Builder() {
        }

        public abstract KeyStore getKeyStore() throws KeyStoreException;

        public abstract ProtectionParameter getProtectionParameter(String alias) throws KeyStoreException;

        public static Builder newInstance(final KeyStore keyStore, final ProtectionParameter protectionParameter);

        public static Builder newInstance(String type, @Nullable Provider provider, File file, ProtectionParameter protection);

        public static Builder newInstance(File file, ProtectionParameter protection);

        private static final class FileBuilder extends Builder {

            @SuppressWarnings("removal")
            public synchronized KeyStore getKeyStore() throws KeyStoreException;

            public synchronized ProtectionParameter getProtectionParameter(String alias);
        }

        public static Builder newInstance(final String type, @Nullable final Provider provider, final ProtectionParameter protection);
    }

    static class SimpleLoadStoreParameter implements LoadStoreParameter {

        public ProtectionParameter getProtectionParameter();
    }
}
