/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.security;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import java.util.*;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import javax.security.auth.DestroyFailedException;
    @Positive
import javax.security.auth.callback.*;
    @Positive
import sun.security.util.Debug;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class KeyStore {

    @Positive
    public static interface LoadStoreParameter {

    @Positive
        @Nullable
    @Positive
        public ProtectionParameter getProtectionParameter();
    @Positive
    }

    @Positive
    public static interface ProtectionParameter {
    @Positive
    }

    @Positive
    public static class PasswordProtection implements ProtectionParameter, javax.security.auth.Destroyable {

    @Positive
        public PasswordProtection(char @Nullable [] password) {
    @Positive
        }

    @Positive
        public PasswordProtection(char @Nullable [] password, String protectionAlgorithm, @Nullable AlgorithmParameterSpec protectionParameters) {
    @Positive
        }

    @Positive
        @Nullable
    @Positive
        public String getProtectionAlgorithm();

    @Positive
        @Nullable
    @Positive
        public AlgorithmParameterSpec getProtectionParameters();

    @Positive
        public synchronized char @Nullable [] getPassword();

    @Positive
        public synchronized void destroy() throws DestroyFailedException;

    @Positive
        public synchronized boolean isDestroyed();
    @Positive
    }

    @Positive
    public static class CallbackHandlerProtection implements ProtectionParameter {

    @Positive
        public CallbackHandlerProtection(CallbackHandler handler) {
    @Positive
        }

    @Positive
        public CallbackHandler getCallbackHandler();
    @Positive
    }

    @Positive
    public static interface Entry {

    @Positive
        public default Set<Attribute> getAttributes();

    @Positive
        public interface Attribute {

    @Positive
            public String getName();

    @Positive
            public String getValue();
    @Positive
        }
    @Positive
    }

    @Positive
    public static final class PrivateKeyEntry implements Entry {

    @Positive
        public PrivateKeyEntry(PrivateKey privateKey, Certificate[] chain) {
    @Positive
        }

    @Positive
        public PrivateKeyEntry(PrivateKey privateKey, Certificate[] chain, Set<Attribute> attributes) {
    @Positive
        }

    @Positive
        public PrivateKey getPrivateKey();

    @Positive
        public Certificate[] getCertificateChain();

    @Positive
        public Certificate getCertificate();

    @Positive
        @Override
    @Positive
        public Set<Attribute> getAttributes();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class SecretKeyEntry implements Entry {

    @Positive
        public SecretKeyEntry(SecretKey secretKey) {
    @Positive
        }

    @Positive
        public SecretKeyEntry(SecretKey secretKey, Set<Attribute> attributes) {
    @Positive
        }

    @Positive
        public SecretKey getSecretKey();

    @Positive
        @Override
    @Positive
        public Set<Attribute> getAttributes();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class TrustedCertificateEntry implements Entry {

    @Positive
        public TrustedCertificateEntry(Certificate trustedCert) {
    @Positive
        }

    @Positive
        public TrustedCertificateEntry(Certificate trustedCert, Set<Attribute> attributes) {
    @Positive
        }

    @Positive
        public Certificate getTrustedCertificate();

    @Positive
        @Override
    @Positive
        public Set<Attribute> getAttributes();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected KeyStore(KeyStoreSpi keyStoreSpi, Provider provider, String type) {
    @Positive
    }

    @Positive
    public static KeyStore getInstance(String type) throws KeyStoreException;

    @Positive
    public static KeyStore getInstance(String type, String provider) throws KeyStoreException, NoSuchProviderException;

    @Positive
    public static KeyStore getInstance(String type, Provider provider) throws KeyStoreException;

    @Positive
    public static final String getDefaultType();

    @Positive
    public final Provider getProvider();

    @Positive
    public final String getType();

    @Positive
    @Nullable
    @Positive
    public final Key getKey(String alias, char[] password) throws KeyStoreException, NoSuchAlgorithmException, UnrecoverableKeyException;

    @Positive
    public final Certificate @Nullable [] getCertificateChain(String alias) throws KeyStoreException;

    @Positive
    @Nullable
    @Positive
    public final Certificate getCertificate(String alias) throws KeyStoreException;

    @Positive
    @Nullable
    @Positive
    public final Date getCreationDate(String alias) throws KeyStoreException;

    @Positive
    public final void setKeyEntry(String alias, Key key, char[] password, Certificate[] chain) throws KeyStoreException;

    @Positive
    public final void setKeyEntry(String alias, byte[] key, Certificate[] chain) throws KeyStoreException;

    @Positive
    public final void setCertificateEntry(String alias, Certificate cert) throws KeyStoreException;

    @Positive
    public final void deleteEntry(String alias) throws KeyStoreException;

    @Positive
    public final Enumeration<String> aliases() throws KeyStoreException;

    @Positive
    @Pure
    @Positive
    public final boolean containsAlias(String alias) throws KeyStoreException;

    @Positive
    public final int size() throws KeyStoreException;

    @Positive
    public final boolean isKeyEntry(String alias) throws KeyStoreException;

    @Positive
    public final boolean isCertificateEntry(String alias) throws KeyStoreException;

    @Positive
    @Nullable
    @Positive
    public final String getCertificateAlias(Certificate cert) throws KeyStoreException;

    @Positive
    public final void store(OutputStream stream, char[] password) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public final void store(@Nullable LoadStoreParameter param) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public final void load(@Nullable InputStream stream, char @Nullable [] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public final void load(@Nullable LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    @Nullable
    @Positive
    public final Entry getEntry(String alias, @Nullable ProtectionParameter protParam) throws NoSuchAlgorithmException, UnrecoverableEntryException, KeyStoreException;

    @Positive
    public final void setEntry(String alias, Entry entry, @Nullable ProtectionParameter protParam) throws KeyStoreException;

    @Positive
    public final boolean entryInstanceOf(String alias, Class<? extends KeyStore.Entry> entryClass) throws KeyStoreException;

    @Positive
    public static final KeyStore getInstance(File file, char @Nullable [] password) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public static final KeyStore getInstance(File file, @Nullable LoadStoreParameter param) throws KeyStoreException, IOException, NoSuchAlgorithmException, CertificateException;

    @Positive
    public abstract static class Builder {

    @Positive
        protected Builder() {
    @Positive
        }

    @Positive
        public abstract KeyStore getKeyStore() throws KeyStoreException;

    @Positive
        public abstract ProtectionParameter getProtectionParameter(String alias) throws KeyStoreException;

    @Positive
        public static Builder newInstance(final KeyStore keyStore, final ProtectionParameter protectionParameter);

    @Positive
        public static Builder newInstance(String type, @Nullable Provider provider, File file, ProtectionParameter protection);

    @Positive
        public static Builder newInstance(File file, ProtectionParameter protection);

    @Positive
        private static final class FileBuilder extends Builder {

    @Positive
            @SuppressWarnings("removal")
    @Positive
            public synchronized KeyStore getKeyStore() throws KeyStoreException;

    @Positive
            public synchronized ProtectionParameter getProtectionParameter(String alias);
    @Positive
        }

    @Positive
        public static Builder newInstance(final String type, @Nullable final Provider provider, final ProtectionParameter protection);
    @Positive
    }

    @Positive
    static class SimpleLoadStoreParameter implements LoadStoreParameter {

    @Positive
        public ProtectionParameter getProtectionParameter();
    @Positive
    }
    @Positive
}
