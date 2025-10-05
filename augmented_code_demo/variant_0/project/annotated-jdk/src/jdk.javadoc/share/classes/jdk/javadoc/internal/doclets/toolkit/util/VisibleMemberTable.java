/*
    @Positive
 * Copyright (c) 2018, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import javax.lang.model.element.AnnotationMirror;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.type.ArrayType;
    @Positive
import javax.lang.model.type.DeclaredType;
    @Positive
import javax.lang.model.type.ExecutableType;
    @Positive
import javax.lang.model.type.TypeKind;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import javax.lang.model.type.WildcardType;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.lang.model.util.SimpleTypeVisitor14;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumMap;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.PropertyUtils;

    @Positive
public class VisibleMemberTable {

    @Positive
    public enum Kind {

    @Positive
        NESTED_CLASSES,
    @Positive
        ENUM_CONSTANTS,
    @Positive
        FIELDS,
    @Positive
        CONSTRUCTORS,
    @Positive
        METHODS,
    @Positive
        ANNOTATION_TYPE_MEMBER_OPTIONAL,
    @Positive
        ANNOTATION_TYPE_MEMBER_REQUIRED,
    @Positive
        PROPERTIES;

    @Positive
        public static Set<Kind> forSummariesOf(ElementKind kind);

    @Positive
        public static Set<Kind> forDetailsOf(ElementKind kind);
    @Positive
    }

    @Positive
    protected VisibleMemberTable(TypeElement typeElement, BaseConfiguration configuration, VisibleMemberCache mcache) {
    @Positive
    }

    @Positive
    List<VisibleMemberTable> getAllSuperclasses();

    @Positive
    List<VisibleMemberTable> getAllSuperinterfaces();

    @Positive
    public List<? extends Element> getAllVisibleMembers(Kind kind);

    @Positive
    public List<? extends Element> getVisibleMembers(Kind kind, Predicate<Element> p);

    @Positive
    public List<? extends Element> getVisibleMembers(Kind kind);

    @Positive
    public List<? extends Element> getMembers(Kind kind);

    @Positive
    public ExecutableElement getOverriddenMethod(ExecutableElement e);

    @Positive
    public ExecutableElement getSimplyOverriddenMethod(ExecutableElement e);

    @Positive
    public Set<TypeElement> getVisibleTypeElements();

    @Positive
    public boolean hasVisibleMembers();

    @Positive
    public boolean hasVisibleMembers(Kind kind);

    @Positive
    public VariableElement getPropertyField(ExecutableElement propertyMethod);

    @Positive
    public ExecutableElement getPropertyGetter(ExecutableElement propertyMethod);

    @Positive
    public ExecutableElement getPropertySetter(ExecutableElement propertyMethod);

    @Positive
    void computeVisibleMembers(LocalMemberTable lmt, Kind kind);

    @Positive
    @Pure
    @Positive
    boolean isEnclosureInterface(Element e);

    @Positive
    boolean allowInheritedMethod(ExecutableElement inheritedMethod, Map<ExecutableElement, List<ExecutableElement>> overriddenByTable, LocalMemberTable lmt);

    @Positive
    class LocalMemberTable {

    @Positive
        String getMemberKey(Element e);

    @Positive
        void addMember(Element e, Kind kind);

    @Positive
        List<Element> getOrderedMembers(Kind kind);

    @Positive
        List<Element> getMembers(Element e, Kind kind);

    @Positive
        List<Element> getMembers(String key, Kind kind);

    @Positive
        List<Element> getPropertyMethods(String methodName, int argcount);
    @Positive
    }

    @Positive
    static class PropertyMembers {

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public List<ExecutableElement> getImplementedMethods(ExecutableElement method);

    @Positive
    public TypeMirror getImplementedMethodHolder(ExecutableElement method, ExecutableElement implementedMethod);

    @Positive
    private class ImplementedMethods {

    @Positive
        public ImplementedMethods(ExecutableElement method) {
    @Positive
        }

    @Positive
        List<ExecutableElement> getImplementedMethods();

    @Positive
        TypeMirror getMethodHolder(ExecutableElement ee);
    @Positive
    }

    @Positive
    static class OverriddenMethodInfo {

    @Positive
        public OverriddenMethodInfo(ExecutableElement overridden, boolean simpleOverride) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
