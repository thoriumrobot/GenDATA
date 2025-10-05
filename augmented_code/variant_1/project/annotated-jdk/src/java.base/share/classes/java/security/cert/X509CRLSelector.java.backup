/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
import java.math.BigInteger;
import java.util.*;
import javax.security.auth.x500.X500Principal;
import sun.security.util.Debug;
import sun.security.util.DerInputStream;
import sun.security.util.KnownOIDs;
import sun.security.x509.CRLNumberExtension;
import sun.security.x509.X500Name;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class X509CRLSelector implements CRLSelector {

    public X509CRLSelector() {
    }

    public void setIssuers(Collection<X500Principal> issuers);

    public void setIssuerNames(Collection<?> names) throws IOException;

    public void addIssuer(X500Principal issuer);

    @Deprecated()
    public void addIssuerName(String name) throws IOException;

    public void addIssuerName(byte[] name) throws IOException;

    public void setMinCRLNumber(BigInteger minCRL);

    public void setMaxCRLNumber(BigInteger maxCRL);

    public void setDateAndTime(Date dateAndTime);

    void setDateAndTime(Date dateAndTime, long skew);

    public void setCertificateChecking(X509Certificate cert);

    public Collection<X500Principal> getIssuers();

    public Collection<Object> getIssuerNames();

    public BigInteger getMinCRL();

    public BigInteger getMaxCRL();

    public Date getDateAndTime();

    public X509Certificate getCertificateChecking();

    public String toString();

    public boolean match(CRL crl);

    public Object clone();
}
