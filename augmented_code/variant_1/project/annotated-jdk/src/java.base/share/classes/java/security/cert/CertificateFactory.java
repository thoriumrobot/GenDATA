/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.security.cert;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.InputStream;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.security.Provider;
    @Positive
import java.security.Security;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.NoSuchProviderException;
    @Positive
import sun.security.jca.*;
    @Positive
import sun.security.jca.GetInstance.Instance;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class CertificateFactory {

    @Positive
    protected CertificateFactory(CertificateFactorySpi certFacSpi, Provider provider, String type) {
    @Positive
    }

    @Positive
    public static final CertificateFactory getInstance(String type) throws CertificateException;

    @Positive
    public static final CertificateFactory getInstance(String type, String provider) throws CertificateException, NoSuchProviderException;

    @Positive
    public static final CertificateFactory getInstance(String type, Provider provider) throws CertificateException;

    @Positive
    public final Provider getProvider();

    @Positive
    public final String getType();

    @Positive
    public final Certificate generateCertificate(InputStream inStream) throws CertificateException;

    @Positive
    public final Iterator<String> getCertPathEncodings();

    @Positive
    public final CertPath generateCertPath(InputStream inStream) throws CertificateException;

    @Positive
    public final CertPath generateCertPath(InputStream inStream, String encoding) throws CertificateException;

    @Positive
    public final CertPath generateCertPath(List<? extends Certificate> certificates) throws CertificateException;

    @Positive
    public final Collection<? extends Certificate> generateCertificates(InputStream inStream) throws CertificateException;

    @Positive
    public final CRL generateCRL(InputStream inStream) throws CRLException;

    @Positive
    public final Collection<? extends CRL> generateCRLs(InputStream inStream) throws CRLException;
    @Positive
}
