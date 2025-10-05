/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.security.cert;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.*;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.DerInputStream;
    @Positive
import sun.security.util.KnownOIDs;
    @Positive
import sun.security.x509.CRLNumberExtension;
    @Positive
import sun.security.x509.X500Name;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class X509CRLSelector implements CRLSelector {

    @Positive
    public X509CRLSelector() {
    @Positive
    }

    @Positive
    public void setIssuers(Collection<X500Principal> issuers);

    @Positive
    public void setIssuerNames(Collection<?> names) throws IOException;

    @Positive
    public void addIssuer(X500Principal issuer);

    @Positive
    @Deprecated()
    @Positive
    public void addIssuerName(String name) throws IOException;

    @Positive
    public void addIssuerName(byte[] name) throws IOException;

    @Positive
    public void setMinCRLNumber(BigInteger minCRL);

    @Positive
    public void setMaxCRLNumber(BigInteger maxCRL);

    @Positive
    public void setDateAndTime(Date dateAndTime);

    @Positive
    void setDateAndTime(Date dateAndTime, long skew);

    @Positive
    public void setCertificateChecking(X509Certificate cert);

    @Positive
    public Collection<X500Principal> getIssuers();

    @Positive
    public Collection<Object> getIssuerNames();

    @Positive
    public BigInteger getMinCRL();

    @Positive
    public BigInteger getMaxCRL();

    @Positive
    public Date getDateAndTime();

    @Positive
    public X509Certificate getCertificateChecking();

    @Positive
    public String toString();

    @Positive
    public boolean match(CRL crl);

    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 1
