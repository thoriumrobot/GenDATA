/*
    @Positive
 * Copyright (c) 2002, 2015, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

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
import java.io.IOException;
    @Positive
import java.io.StringReader;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.*;
    @Positive
import sun.security.util.*;

    @Positive
public class RDN {

    @Positive
    public RDN(String name) throws IOException {
    @Positive
    }

    @Positive
    public RDN(String name, Map<String, String> keywordMap) throws IOException {
    @Positive
    }

    @Positive
    public RDN(AVA ava) {
    @Positive
    }

    @Positive
    public RDN(AVA[] avas) {
    @Positive
    }

    @Positive
    public List<AVA> avas();

    @Positive
    public int size();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    DerValue findAttribute(ObjectIdentifier oid);

    @Positive
    void encode(DerOutputStream out) throws IOException;

    @Positive
    public String toString();

    @Positive
    public String toRFC1779String();

    @Positive
    public String toRFC1779String(Map<String, String> oidMap);

    @Positive
    public String toRFC2253String();

    @Positive
    public String toRFC2253String(Map<String, String> oidMap);

    @Positive
    public String toRFC2253String(boolean canonical);
    @Positive
}

    @Positive
class AVAComparator implements Comparator<AVA> {

    @Positive
    static Comparator<AVA> getInstance();

    @Positive
    public int compare(AVA a1, AVA a2);
    @Positive
}

// CFWR semantic augmentation - variant 0
