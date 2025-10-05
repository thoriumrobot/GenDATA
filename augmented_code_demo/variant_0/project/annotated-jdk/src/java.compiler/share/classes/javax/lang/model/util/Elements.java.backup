/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.lang.model.util;

import org.checkerframework.checker.signature.qual.CanonicalName;
import org.checkerframework.checker.signature.qual.FullyQualifiedName;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashSet;
import java.util.Objects;
import javax.lang.model.AnnotatedConstruct;
import javax.lang.model.element.*;

public interface Elements {

    PackageElement getPackageElement(CharSequence name);

    default PackageElement getPackageElement(ModuleElement module, CharSequence name);

    default Set<? extends PackageElement> getAllPackageElements(@FullyQualifiedName CharSequence name);

    TypeElement getTypeElement(@FullyQualifiedName CharSequence name);

    default TypeElement getTypeElement(ModuleElement module, @FullyQualifiedName CharSequence name);

    default Set<? extends TypeElement> getAllTypeElements(@FullyQualifiedName CharSequence name);

    default ModuleElement getModuleElement(CharSequence name);

    default Set<? extends ModuleElement> getAllModuleElements();

    Map<? extends ExecutableElement, ? extends AnnotationValue> getElementValuesWithDefaults(AnnotationMirror a);

    String getDocComment(Element e);

    boolean isDeprecated(Element e);

    default Origin getOrigin(Element e);

    default Origin getOrigin(AnnotatedConstruct c, AnnotationMirror a);

    default Origin getOrigin(ModuleElement m, ModuleElement.Directive directive);

    public enum Origin {

        EXPLICIT, MANDATED, SYNTHETIC;

        public boolean isDeclared();
    }

    default boolean isBridge(ExecutableElement e);

    Name getBinaryName(TypeElement type);

    PackageElement getPackageOf(Element e);

    default ModuleElement getModuleOf(Element e);

    List<? extends Element> getAllMembers(TypeElement type);

    List<? extends AnnotationMirror> getAllAnnotationMirrors(Element e);

    boolean hides(Element hider, Element hidden);

    boolean overrides(ExecutableElement overrider, ExecutableElement overridden, TypeElement type);

    String getConstantExpression(Object value);

    void printElements(java.io.Writer w, Element... elements);

    Name getName(CharSequence cs);

    boolean isFunctionalInterface(TypeElement type);

    default boolean isAutomaticModule(ModuleElement module);

    default RecordComponentElement recordComponentFor(ExecutableElement accessor);
}
