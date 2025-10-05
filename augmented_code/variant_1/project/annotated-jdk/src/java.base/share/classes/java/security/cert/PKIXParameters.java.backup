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
import java.security.InvalidAlgorithmParameterException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class PKIXParameters implements CertPathParameters {

    public PKIXParameters(Set<TrustAnchor> trustAnchors) throws InvalidAlgorithmParameterException {
    }

    public PKIXParameters(KeyStore keystore) throws KeyStoreException, InvalidAlgorithmParameterException {
    }

    public Set<TrustAnchor> getTrustAnchors();

    public void setTrustAnchors(Set<TrustAnchor> trustAnchors) throws InvalidAlgorithmParameterException;

    public Set<String> getInitialPolicies();

    public void setInitialPolicies(Set<String> initialPolicies);

    public void setCertStores(List<CertStore> stores);

    public void addCertStore(CertStore store);

    public List<CertStore> getCertStores();

    public void setRevocationEnabled(boolean val);

    public boolean isRevocationEnabled();

    public void setExplicitPolicyRequired(boolean val);

    public boolean isExplicitPolicyRequired();

    public void setPolicyMappingInhibited(boolean val);

    public boolean isPolicyMappingInhibited();

    public void setAnyPolicyInhibited(boolean val);

    public boolean isAnyPolicyInhibited();

    public void setPolicyQualifiersRejected(boolean qualifiersRejected);

    public boolean getPolicyQualifiersRejected();

    public Date getDate();

    public void setDate(Date date);

    public void setCertPathCheckers(List<PKIXCertPathChecker> checkers);

    public List<PKIXCertPathChecker> getCertPathCheckers();

    public void addCertPathChecker(PKIXCertPathChecker checker);

    public String getSigProvider();

    public void setSigProvider(String sigProvider);

    public CertSelector getTargetCertConstraints();

    public void setTargetCertConstraints(CertSelector selector);

    public Object clone();

    public String toString();
}
