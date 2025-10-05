/*
    @Positive
 * Copyright (c) 2000, 2013, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.jgss;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.ietf.jgss.*;
    @Positive
import sun.security.jgss.spi.*;
    @Positive
import java.util.*;
    @Positive
import sun.security.jgss.spnego.SpNegoCredElement;

    @Positive
public class GSSCredentialImpl implements GSSCredential {

    @Positive
    public GSSCredentialImpl() {
    @Positive
    }

    @Positive
    protected GSSCredentialImpl(GSSCredentialImpl src) {
    @Positive
    }

    @Positive
    public GSSCredentialImpl(GSSManagerImpl gssManager, GSSCredentialSpi mechElement) throws GSSException {
    @Positive
    }

    @Positive
    void init(GSSManagerImpl gssManager);

    @Positive
    public void dispose() throws GSSException;

    @Positive
    public GSSCredential impersonate(GSSName name) throws GSSException;

    @Positive
    public GSSName getName() throws GSSException;

    @Positive
    public GSSName getName(Oid mech) throws GSSException;

    @Positive
    public int getRemainingLifetime() throws GSSException;

    @Positive
    public int getRemainingInitLifetime(Oid mech) throws GSSException;

    @Positive
    public int getRemainingAcceptLifetime(Oid mech) throws GSSException;

    @Positive
    public int getUsage() throws GSSException;

    @Positive
    public int getUsage(Oid mech) throws GSSException;

    @Positive
    public Oid[] getMechs() throws GSSException;

    @Positive
    public void add(GSSName name, int initLifetime, int acceptLifetime, Oid mech, int usage) throws GSSException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object another);

    @Positive
    public int hashCode();

    @Positive
    public GSSCredentialSpi getElement(Oid mechOid, boolean initiate) throws GSSException;

    @Positive
    Set<GSSCredentialSpi> getElements();

    @Positive
    public String toString();

    @Positive
    static class SearchKey {

    @Positive
        public SearchKey(Oid mechOid, int usage) {
    @Positive
        }

    @Positive
        public Oid getMech();

    @Positive
        public int getUsage();

    @Positive
        public boolean equals(Object other);

    @Positive
        public int hashCode();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
