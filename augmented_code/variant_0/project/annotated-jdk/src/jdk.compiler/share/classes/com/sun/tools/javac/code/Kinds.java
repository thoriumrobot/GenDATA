/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.Set;
    @Positive
import java.util.Locale;
    @Positive
import com.sun.source.tree.MemberReferenceTree;
    @Positive
import com.sun.tools.javac.api.Formattable;
    @Positive
import com.sun.tools.javac.api.Messages;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.TypeTag.CLASS;
    @Positive
import static com.sun.tools.javac.code.TypeTag.PACKAGE;
    @Positive
import static com.sun.tools.javac.code.TypeTag.TYPEVAR;

    @Positive
public class Kinds {

    @Positive
    public enum Kind {

    @Positive
        NIL(Category.BASIC, KindSelector.NIL),
    @Positive
        PCK(Category.BASIC, KindName.PACKAGE, KindSelector.PCK),
    @Positive
        TYP(Category.BASIC, KindName.CLASS, KindSelector.TYP),
    @Positive
        VAR(Category.BASIC, KindName.VAR, KindSelector.VAR),
    @Positive
        MTH(Category.BASIC, KindName.METHOD, KindSelector.MTH),
    @Positive
        POLY(Category.BASIC, KindSelector.POLY),
    @Positive
        MDL(Category.BASIC, KindSelector.MDL),
    @Positive
        ERR(Category.ERROR, KindSelector.ERR),
    @Positive
        AMBIGUOUS(Category.RESOLUTION_TARGET),
    @Positive
        HIDDEN(Category.RESOLUTION_TARGET),
    @Positive
        STATICERR(Category.RESOLUTION_TARGET),
    @Positive
        MISSING_ENCL(Category.RESOLUTION),
    @Positive
        BAD_RESTRICTED_TYPE(Category.RESOLUTION),
    @Positive
        ABSENT_VAR(Category.RESOLUTION_TARGET, KindName.VAR),
    @Positive
        WRONG_MTHS(Category.RESOLUTION_TARGET, KindName.METHOD),
    @Positive
        WRONG_MTH(Category.RESOLUTION_TARGET, KindName.METHOD),
    @Positive
        ABSENT_MTH(Category.RESOLUTION_TARGET, KindName.METHOD),
    @Positive
        ABSENT_TYP(Category.RESOLUTION_TARGET, KindName.CLASS);

    @Positive
        public KindSelector toSelector();

    @Positive
        public boolean matches(KindSelector kindSelectors);

    @Positive
        public boolean isResolutionError();

    @Positive
        public boolean isResolutionTargetError();

    @Positive
        public boolean isValid();

    @Positive
        public boolean betterThan(Kind other);

    @Positive
        public KindName kindName();

    @Positive
        public KindName absentKind();
    @Positive
    }

    @Positive
    public static class KindSelector {

    @Positive
        public static final KindSelector NIL;

    @Positive
        public static final KindSelector PCK;

    @Positive
        public static final KindSelector TYP;

    @Positive
        public static final KindSelector VAR;

    @Positive
        public static final KindSelector VAL;

    @Positive
        public static final KindSelector MTH;

    @Positive
        public static final KindSelector POLY;

    @Positive
        public static final KindSelector MDL;

    @Positive
        public static final KindSelector ERR;

    @Positive
        public static final KindSelector ASG;

    @Positive
        public static final KindSelector TYP_PCK;

    @Positive
        public static final KindSelector VAL_MTH;

    @Positive
        public static final KindSelector VAL_POLY;

    @Positive
        public static final KindSelector VAL_TYP;

    @Positive
        public static final KindSelector VAL_TYP_PCK;

    @Positive
        public static KindSelector of(KindSelector... kindSelectors);

    @Positive
        public boolean subset(KindSelector other);

    @Positive
        @Pure
    @Positive
        public boolean contains(KindSelector other);

    @Positive
        public Set<KindName> kindNames();
    @Positive
    }

    @Positive
    public enum KindName implements Formattable {

    @Positive
        ANNOTATION("kindname.annotation"),
    @Positive
        CONSTRUCTOR("kindname.constructor"),
    @Positive
        INTERFACE("kindname.interface"),
    @Positive
        ENUM("kindname.enum"),
    @Positive
        STATIC("kindname.static"),
    @Positive
        TYPEVAR("kindname.type.variable"),
    @Positive
        BOUND("kindname.type.variable.bound"),
    @Positive
        VAR("kindname.variable"),
    @Positive
        VAL("kindname.value"),
    @Positive
        METHOD("kindname.method"),
    @Positive
        CLASS("kindname.class"),
    @Positive
        STATIC_INIT("kindname.static.init"),
    @Positive
        INSTANCE_INIT("kindname.instance.init"),
    @Positive
        PACKAGE("kindname.package"),
    @Positive
        MODULE("kindname.module"),
    @Positive
        RECORD_COMPONENT("kindname.record.component"),
    @Positive
        RECORD("kindname.record");

    @Positive
        public String toString();

    @Positive
        public String getKind();

    @Positive
        public String toString(Locale locale, Messages messages);
    @Positive
    }

    @Positive
    public static KindName kindName(MemberReferenceTree.ReferenceMode mode);

    @Positive
    public static KindName kindName(Symbol sym);

    @Positive
    public static KindName typeKindName(Type t);
    @Positive
}
