/*
 * Copyright (c) 1999, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

import org.checkerframework.dataflow.qual.Pure;
import java.util.EnumSet;
import java.util.Set;
import java.util.Locale;
import com.sun.source.tree.MemberReferenceTree;
import com.sun.tools.javac.api.Formattable;
import com.sun.tools.javac.api.Messages;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.TypeTag.CLASS;
import static com.sun.tools.javac.code.TypeTag.PACKAGE;
import static com.sun.tools.javac.code.TypeTag.TYPEVAR;

public class Kinds {

    public enum Kind {

        NIL(Category.BASIC, KindSelector.NIL),
        PCK(Category.BASIC, KindName.PACKAGE, KindSelector.PCK),
        TYP(Category.BASIC, KindName.CLASS, KindSelector.TYP),
        VAR(Category.BASIC, KindName.VAR, KindSelector.VAR),
        MTH(Category.BASIC, KindName.METHOD, KindSelector.MTH),
        POLY(Category.BASIC, KindSelector.POLY),
        MDL(Category.BASIC, KindSelector.MDL),
        ERR(Category.ERROR, KindSelector.ERR),
        AMBIGUOUS(Category.RESOLUTION_TARGET),
        HIDDEN(Category.RESOLUTION_TARGET),
        STATICERR(Category.RESOLUTION_TARGET),
        MISSING_ENCL(Category.RESOLUTION),
        BAD_RESTRICTED_TYPE(Category.RESOLUTION),
        ABSENT_VAR(Category.RESOLUTION_TARGET, KindName.VAR),
        WRONG_MTHS(Category.RESOLUTION_TARGET, KindName.METHOD),
        WRONG_MTH(Category.RESOLUTION_TARGET, KindName.METHOD),
        ABSENT_MTH(Category.RESOLUTION_TARGET, KindName.METHOD),
        ABSENT_TYP(Category.RESOLUTION_TARGET, KindName.CLASS);

        public KindSelector toSelector();

        public boolean matches(KindSelector kindSelectors);

        public boolean isResolutionError();

        public boolean isResolutionTargetError();

        public boolean isValid();

        public boolean betterThan(Kind other);

        public KindName kindName();

        public KindName absentKind();
    }

    public static class KindSelector {

        public static final KindSelector NIL;

        public static final KindSelector PCK;

        public static final KindSelector TYP;

        public static final KindSelector VAR;

        public static final KindSelector VAL;

        public static final KindSelector MTH;

        public static final KindSelector POLY;

        public static final KindSelector MDL;

        public static final KindSelector ERR;

        public static final KindSelector ASG;

        public static final KindSelector TYP_PCK;

        public static final KindSelector VAL_MTH;

        public static final KindSelector VAL_POLY;

        public static final KindSelector VAL_TYP;

        public static final KindSelector VAL_TYP_PCK;

        public static KindSelector of(KindSelector... kindSelectors);

        public boolean subset(KindSelector other);

        @Pure
        public boolean contains(KindSelector other);

        public Set<KindName> kindNames();
    }

    public enum KindName implements Formattable {

        ANNOTATION("kindname.annotation"),
        CONSTRUCTOR("kindname.constructor"),
        INTERFACE("kindname.interface"),
        ENUM("kindname.enum"),
        STATIC("kindname.static"),
        TYPEVAR("kindname.type.variable"),
        BOUND("kindname.type.variable.bound"),
        VAR("kindname.variable"),
        VAL("kindname.value"),
        METHOD("kindname.method"),
        CLASS("kindname.class"),
        STATIC_INIT("kindname.static.init"),
        INSTANCE_INIT("kindname.instance.init"),
        PACKAGE("kindname.package"),
        MODULE("kindname.module"),
        RECORD_COMPONENT("kindname.record.component"),
        RECORD("kindname.record");

        public String toString();

        public String getKind();

        public String toString(Locale locale, Messages messages);
    }

    public static KindName kindName(MemberReferenceTree.ReferenceMode mode);

    public static KindName kindName(Symbol sym);

    public static KindName typeKindName(Type t);
}
