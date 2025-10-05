/*
 * Copyright (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.security.cert;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.InputStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.security.Provider;
import java.security.Security;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import sun.security.jca.*;
import sun.security.jca.GetInstance.Instance;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class CertificateFactory {

    protected CertificateFactory(CertificateFactorySpi certFacSpi, Provider provider, String type) {
    }

    public static final CertificateFactory getInstance(String type) throws CertificateException;

    public static final CertificateFactory getInstance(String type, String provider) throws CertificateException, NoSuchProviderException;

    public static final CertificateFactory getInstance(String type, Provider provider) throws CertificateException;

    public final Provider getProvider();

    public final String getType();

    public final Certificate generateCertificate(InputStream inStream) throws CertificateException;

    public final Iterator<String> getCertPathEncodings();

    public final CertPath generateCertPath(InputStream inStream) throws CertificateException;

    public final CertPath generateCertPath(InputStream inStream, String encoding) throws CertificateException;

    public final CertPath generateCertPath(List<? extends Certificate> certificates) throws CertificateException;

    public final Collection<? extends Certificate> generateCertificates(InputStream inStream) throws CertificateException;

    public final CRL generateCRL(InputStream inStream) throws CRLException;

    public final Collection<? extends CRL> generateCRLs(InputStream inStream) throws CRLException;
}
