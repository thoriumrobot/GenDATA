/*
    @Positive
 * Copyright (c) 2014, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac.pubapi;

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
import static com.sun.tools.sjavac.Util.union;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import com.sun.tools.javac.util.Assert;
    @Positive
import com.sun.tools.javac.util.StringUtils;

    @Positive
public class PubApi implements Serializable {

    @Positive
    public final Map<String, PubType> types;

    @Positive
    public final Map<String, PubVar> variables;

    @Positive
    public final Map<String, PubMethod> methods;

    @Positive
    public final Map<String, PubVar> recordComponents;

    @Positive
    public PubApi() {
    @Positive
    }

    @Positive
    public PubApi(Collection<PubType> types, Collection<PubVar> variables, Collection<PubMethod> methods) {
    @Positive
    }

    @Positive
    public boolean isBackwardCompatibleWith(PubApi older);

    @Positive
    public List<String> asListOfStrings();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    public static PubApi mergeTypes(PubApi api1, PubApi api2);

    @Positive
    public void appendItem(String l);

    @Positive
    public void addPubType(PubType t);

    @Positive
    public void addPubVar(PubVar v);

    @Positive
    public void addPubMethod(PubMethod m);

    @Positive
    public Set<Modifier> parseModifiers(String modifiers);

    @Positive
    public List<String> splitOnTopLevelCommas(String s);

    @Positive
    public static List<String> splitOnTopLevelChars(String s, char split);

    @Positive
    public boolean isEmpty();

    @Positive
    public List<String> diff(PubApi prevApi);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
