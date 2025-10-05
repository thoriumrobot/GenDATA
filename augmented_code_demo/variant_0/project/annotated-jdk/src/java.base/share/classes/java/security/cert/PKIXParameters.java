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
import java.security.InvalidAlgorithmParameterException;
    @Positive
import java.security.KeyStore;
    @Positive
import java.security.KeyStoreException;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Date;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class PKIXParameters implements CertPathParameters {

    @Positive
    public PKIXParameters(Set<TrustAnchor> trustAnchors) throws InvalidAlgorithmParameterException {
    @Positive
    }

    @Positive
    public PKIXParameters(KeyStore keystore) throws KeyStoreException, InvalidAlgorithmParameterException {
    @Positive
    }

    @Positive
    public Set<TrustAnchor> getTrustAnchors();

    @Positive
    public void setTrustAnchors(Set<TrustAnchor> trustAnchors) throws InvalidAlgorithmParameterException;

    @Positive
    public Set<String> getInitialPolicies();

    @Positive
    public void setInitialPolicies(Set<String> initialPolicies);

    @Positive
    public void setCertStores(List<CertStore> stores);

    @Positive
    public void addCertStore(CertStore store);

    @Positive
    public List<CertStore> getCertStores();

    @Positive
    public void setRevocationEnabled(boolean val);

    @Positive
    public boolean isRevocationEnabled();

    @Positive
    public void setExplicitPolicyRequired(boolean val);

    @Positive
    public boolean isExplicitPolicyRequired();

    @Positive
    public void setPolicyMappingInhibited(boolean val);

    @Positive
    public boolean isPolicyMappingInhibited();

    @Positive
    public void setAnyPolicyInhibited(boolean val);

    @Positive
    public boolean isAnyPolicyInhibited();

    @Positive
    public void setPolicyQualifiersRejected(boolean qualifiersRejected);

    @Positive
    public boolean getPolicyQualifiersRejected();

    @Positive
    public Date getDate();

    @Positive
    public void setDate(Date date);

    @Positive
    public void setCertPathCheckers(List<PKIXCertPathChecker> checkers);

    @Positive
    public List<PKIXCertPathChecker> getCertPathCheckers();

    @Positive
    public void addCertPathChecker(PKIXCertPathChecker checker);

    @Positive
    public String getSigProvider();

    @Positive
    public void setSigProvider(String sigProvider);

    @Positive
    public CertSelector getTargetCertConstraints();

    @Positive
    public void setTargetCertConstraints(CertSelector selector);

    @Positive
    public Object clone();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
