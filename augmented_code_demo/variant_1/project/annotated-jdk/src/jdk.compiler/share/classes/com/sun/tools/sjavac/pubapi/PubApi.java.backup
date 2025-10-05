/*
 * Copyright (c) 2014, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac.pubapi;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static com.sun.tools.sjavac.Util.union;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.lang.model.element.Modifier;
import com.sun.tools.javac.util.Assert;
import com.sun.tools.javac.util.StringUtils;

public class PubApi implements Serializable {

    public final Map<String, PubType> types;

    public final Map<String, PubVar> variables;

    public final Map<String, PubMethod> methods;

    public final Map<String, PubVar> recordComponents;

    public PubApi() {
    }

    public PubApi(Collection<PubType> types, Collection<PubVar> variables, Collection<PubMethod> methods) {
    }

    public boolean isBackwardCompatibleWith(PubApi older);

    public List<String> asListOfStrings();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    public static PubApi mergeTypes(PubApi api1, PubApi api2);

    public void appendItem(String l);

    public void addPubType(PubType t);

    public void addPubVar(PubVar v);

    public void addPubMethod(PubMethod m);

    public Set<Modifier> parseModifiers(String modifiers);

    public List<String> splitOnTopLevelCommas(String s);

    public static List<String> splitOnTopLevelChars(String s, char split);

    public boolean isEmpty();

    public List<String> diff(PubApi prevApi);

    public String toString();
}
