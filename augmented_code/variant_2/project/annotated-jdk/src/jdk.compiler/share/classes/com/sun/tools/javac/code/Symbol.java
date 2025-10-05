/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.checker.signature.qual.CanonicalName;
    @Positive
import org.checkerframework.checker.interning.qual.InternedDistinct;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.annotation.Inherited;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.Callable;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.function.Predicate;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.lang.model.element.ElementVisitor;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.NestingKind;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.RecordComponentElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.TypeParameterElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import com.sun.tools.javac.code.Kinds.Kind;
    @Positive
import com.sun.tools.javac.comp.Annotate.AnnotationTypeMetadata;
    @Positive
import com.sun.tools.javac.code.Type.*;
    @Positive
import com.sun.tools.javac.comp.Attr;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.jvm.*;
    @Positive
import com.sun.tools.javac.jvm.PoolConstant;
    @Positive
import com.sun.tools.javac.tree.JCTree;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCAnnotation;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCFieldAccess;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCVariableDecl;
    @Positive
import com.sun.tools.javac.tree.JCTree.Tag;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.List;
    @Positive
import com.sun.tools.javac.util.Name;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
    @Positive
import com.sun.tools.javac.code.Scope.WriteableScope;
    @Positive
import static com.sun.tools.javac.code.TypeTag.CLASS;
    @Positive
import static com.sun.tools.javac.code.TypeTag.FORALL;
    @Positive
import static com.sun.tools.javac.code.TypeTag.TYPEVAR;
    @Positive
import static com.sun.tools.javac.jvm.ByteCodes.iadd;
    @Positive
import static com.sun.tools.javac.jvm.ByteCodes.ishll;
    @Positive
import static com.sun.tools.javac.jvm.ByteCodes.lushrl;
    @Positive
import static com.sun.tools.javac.jvm.ByteCodes.lxor;
    @Positive
import static com.sun.tools.javac.jvm.ByteCodes.string_add;

    @Positive
public abstract class Symbol extends AnnoConstruct implements PoolConstant, Element {

    @Positive
    public Kind kind;

    @Positive
    public long flags_field;

    @Positive
    public long flags();

    @Positive
    public Name name;

    @Positive
    public Type type;

    @Positive
    public Symbol owner;

    @Positive
    public Completer completer;

    @Positive
    public Type erasure_field;

    @Positive
    protected SymbolMetadata metadata;

    @Positive
    public List<Attribute.Compound> getRawAttributes();

    @Positive
    public List<Attribute.TypeCompound> getRawTypeAttributes();

    @Positive
    public Attribute.Compound attribute(Symbol anno);

    @Positive
    public boolean annotationsPendingCompletion();

    @Positive
    public void appendAttributes(List<Attribute.Compound> l);

    @Positive
    public void appendClassInitTypeAttributes(List<Attribute.TypeCompound> l);

    @Positive
    public void appendInitTypeAttributes(List<Attribute.TypeCompound> l);

    @Positive
    public void appendUniqueTypeAttributes(List<Attribute.TypeCompound> l);

    @Positive
    public List<Attribute.TypeCompound> getClassInitTypeAttributes();

    @Positive
    public List<Attribute.TypeCompound> getInitTypeAttributes();

    @Positive
    public void setInitTypeAttributes(List<Attribute.TypeCompound> l);

    @Positive
    public void setClassInitTypeAttributes(List<Attribute.TypeCompound> l);

    @Positive
    public List<Attribute.Compound> getDeclarationAttributes();

    @Positive
    public boolean hasAnnotations();

    @Positive
    public boolean hasTypeAnnotations();

    @Positive
    public boolean isCompleted();

    @Positive
    public void prependAttributes(List<Attribute.Compound> l);

    @Positive
    public void resetAnnotations();

    @Positive
    public void setAttributes(Symbol other);

    @Positive
    public void setDeclarationAttributes(List<Attribute.Compound> a);

    @Positive
    public void setTypeAttributes(List<Attribute.TypeCompound> a);

    @Positive
    public SymbolMetadata getMetadata();

    @Positive
    public Symbol(Kind kind, long flags, Name name, Type type, Symbol owner) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public int poolTag();

    @Positive
    public Symbol clone(Symbol newOwner);

    @Positive
    public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
    @CanonicalName
    @Positive
    public String toString();

    @Positive
    public Symbol location();

    @Positive
    public Symbol location(Type site, Types types);

    @Positive
    public Symbol baseSymbol();

    @Positive
    public Type erasure(Types types);

    @Positive
    public Type externalType(Types types);

    @Positive
    public boolean isDeprecated();

    @Positive
    public boolean hasDeprecatedAnnotation();

    @Positive
    public boolean isDeprecatedForRemoval();

    @Positive
    public boolean isPreviewApi();

    @Positive
    public boolean isDeprecatableViaAnnotation();

    @Positive
    public boolean isStatic();

    @Positive
    public boolean isInterface();

    @Positive
    public boolean isAbstract();

    @Positive
    public boolean isPrivate();

    @Positive
    public boolean isPublic();

    @Positive
    public boolean isEnum();

    @Positive
    public boolean isSealed();

    @Positive
    public boolean isNonSealed();

    @Positive
    public boolean isFinal();

    @Positive
    public boolean isDirectlyOrIndirectlyLocal();

    @Positive
    public boolean isAnonymous();

    @Positive
    public boolean isConstructor();

    @Positive
    public boolean isDynamic();

    @Positive
    @CanonicalName
    @Positive
    public Name getQualifiedName();

    @Positive
    @BinaryName
    @Positive
    public Name flatName();

    @Positive
    public WriteableScope members();

    @Positive
    public boolean isInner();

    @Positive
    public boolean hasOuterInstance();

    @Positive
    public ClassSymbol enclClass();

    @Positive
    public ClassSymbol outermostClass();

    @Positive
    public PackageSymbol packge();

    @Positive
    public boolean isSubClass(Symbol base, Types types);

    @Positive
    public boolean isMemberOf(TypeSymbol clazz, Types types);

    @Positive
    public boolean isEnclosedBy(ClassSymbol clazz);

    @Positive
    public final boolean isAccessibleIn(Symbol clazz, Types types);

    @Positive
    public boolean isInheritedIn(Symbol clazz, Types types);

    @Positive
    public Symbol asMemberOf(Type site, Types types);

    @Positive
    public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult);

    @Positive
    public void complete() throws CompletionFailure;

    @Positive
    public void apiComplete() throws CompletionFailure;

    @Positive
    public boolean exists();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public Type asType();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public Symbol getEnclosingElement();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public ElementKind getKind();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public Set<Modifier> getModifiers();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public Name getSimpleName();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public List<Attribute.Compound> getAnnotationMirrors();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public java.util.List<Symbol> getEnclosedElements();

    @Positive
    public List<TypeVariableSymbol> getTypeParameters();

    @Positive
    public static class DelegatedSymbol<T extends Symbol> extends Symbol {

    @Positive
        protected T other;

    @Positive
        public DelegatedSymbol(T other) {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        public Symbol location();

    @Positive
        public Symbol location(Type site, Types types);

    @Positive
        public Symbol baseSymbol();

    @Positive
        public Type erasure(Types types);

    @Positive
        public Type externalType(Types types);

    @Positive
        public boolean isDirectlyOrIndirectlyLocal();

    @Positive
        public boolean isConstructor();

    @Positive
        @CanonicalName
    @Positive
        public Name getQualifiedName();

    @Positive
        @BinaryName
    @Positive
        public Name flatName();

    @Positive
        public WriteableScope members();

    @Positive
        public boolean isInner();

    @Positive
        public boolean hasOuterInstance();

    @Positive
        public ClassSymbol enclClass();

    @Positive
        public ClassSymbol outermostClass();

    @Positive
        public PackageSymbol packge();

    @Positive
        public boolean isSubClass(Symbol base, Types types);

    @Positive
        public boolean isMemberOf(TypeSymbol clazz, Types types);

    @Positive
        public boolean isEnclosedBy(ClassSymbol clazz);

    @Positive
        public boolean isInheritedIn(Symbol clazz, Types types);

    @Positive
        public Symbol asMemberOf(Type site, Types types);

    @Positive
        public void complete() throws CompletionFailure;

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
        public T getUnderlyingSymbol();
    @Positive
    }

    @Positive
    public static abstract class TypeSymbol extends Symbol {

    @Positive
        public TypeSymbol(Kind kind, long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        @CanonicalName
    @Positive
        public static Name formFullName(Name name, Symbol owner);

    @Positive
        @BinaryName
    @Positive
        public static Name formFlatName(Name name, Symbol owner);

    @Positive
        public final boolean precedes(TypeSymbol that, Types types);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Symbol> getEnclosedElements();

    @Positive
        public AnnotationTypeMetadata getAnnotationTypeMetadata();

    @Positive
        public boolean isAnnotationType();

    @Positive
        @Override
    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class TypeVariableSymbol extends TypeSymbol implements TypeParameterElement {

    @Positive
        public TypeVariableSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Symbol getGenericElement();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getBounds();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Attribute.Compound> getAnnotationMirrors();

    @Positive
        @Override
    @Positive
        public <A extends Annotation> Attribute.Compound getAttribute(Class<A> annoType);

    @Positive
        boolean isCurrentSymbolsAnnotation(Attribute.TypeCompound anno, int index);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ModuleSymbol extends TypeSymbol implements ModuleElement {

    @Positive
        public Name version;

    @Positive
        public JavaFileManager.Location sourceLocation;

    @Positive
        public JavaFileManager.Location classLocation;

    @Positive
        public JavaFileManager.Location patchLocation;

    @Positive
        public JavaFileManager.Location patchOutputLocation;

    @Positive
        public List<com.sun.tools.javac.code.Directive> directives;

    @Positive
        public List<com.sun.tools.javac.code.Directive.RequiresDirective> requires;

    @Positive
        public List<com.sun.tools.javac.code.Directive.ExportsDirective> exports;

    @Positive
        public List<com.sun.tools.javac.code.Directive.OpensDirective> opens;

    @Positive
        public List<com.sun.tools.javac.code.Directive.ProvidesDirective> provides;

    @Positive
        public List<com.sun.tools.javac.code.Directive.UsesDirective> uses;

    @Positive
        public ClassSymbol module_info;

    @Positive
        public PackageSymbol unnamedPackage;

    @Positive
        public Map<Name, PackageSymbol> visiblePackages;

    @Positive
        public Set<ModuleSymbol> readModules;

    @Positive
        public List<Symbol> enclosedPackages;

    @Positive
        public Completer usesProvidesCompleter;

    @Positive
        public final Set<ModuleFlags> flags;

    @Positive
        public final Set<ModuleResolutionFlags> resolutionFlags;

    @Positive
        public static ModuleSymbol create(Name name, Name module_info);

    @Positive
        public ModuleSymbol(Name name, Symbol owner) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Name getSimpleName();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean isOpen();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean isUnnamed();

    @Positive
        @Override
    @Positive
        public boolean isDeprecated();

    @Positive
        public boolean isNoModule();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public java.util.List<Directive> getDirectives();

    @Positive
        public void completeUsesProvides();

    @Positive
        @Override
    @Positive
        public ClassSymbol outermostClass();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Symbol> getEnclosedElements();

    @Positive
        public void reset();
    @Positive
    }

    @Positive
    public enum ModuleFlags {

    @Positive
        OPEN(0x0020), SYNTHETIC(0x1000), MANDATED(0x8000);

    @Positive
        public static int value(Set<ModuleFlags> s);

    @Positive
        public final int value;
    @Positive
    }

    @Positive
    public enum ModuleResolutionFlags {

    @Positive
        DO_NOT_RESOLVE_BY_DEFAULT(0x0001), WARN_DEPRECATED(0x0002), WARN_DEPRECATED_REMOVAL(0x0004), WARN_INCUBATING(0x0008);

    @Positive
        public static int value(Set<ModuleResolutionFlags> s);

    @Positive
        public final int value;
    @Positive
    }

    @Positive
    public static class PackageSymbol extends TypeSymbol implements PackageElement {

    @Positive
        public WriteableScope members_field;

    @Positive
        public Name fullname;

    @Positive
        public ClassSymbol package_info;

    @Positive
        public ModuleSymbol modle;

    @Positive
        public JavaFileObject sourcefile;

    @Positive
        public PackageSymbol(Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        public PackageSymbol(Name name, Symbol owner) {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        @CanonicalName
    @Positive
        public Name getQualifiedName();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean isUnnamed();

    @Positive
        public WriteableScope members();

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        public long flags();

    @Positive
        @Override
    @Positive
        public List<Attribute.Compound> getRawAttributes();

    @Positive
        public boolean exists();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Symbol getEnclosingElement();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
        public void reset();
    @Positive
    }

    @Positive
    public static class RootPackageSymbol extends PackageSymbol {

    @Positive
        public final MissingInfoHandler missingInfoHandler;

    @Positive
        public final boolean allowPrivateInvokeVirtual;

    @Positive
        public RootPackageSymbol(Name name, Symbol owner, MissingInfoHandler missingInfoHandler, boolean allowPrivateInvokeVirtual) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static class ClassSymbol extends TypeSymbol implements TypeElement {

    @Positive
        public WriteableScope members_field;

    @Positive
        public Name fullname;

    @Positive
        @BinaryName
    @Positive
        public Name flatname;

    @Positive
        public JavaFileObject sourcefile;

    @Positive
        public JavaFileObject classfile;

    @Positive
        public List<ClassSymbol> trans_local;

    @Positive
        public List<Symbol> permitted;

    @Positive
        public boolean isPermittedExplicit;

    @Positive
        public ClassSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        public ClassSymbol(long flags, Name name, Symbol owner) {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        public long flags();

    @Positive
        public WriteableScope members();

    @Positive
        @Override
    @Positive
        public List<Attribute.Compound> getRawAttributes();

    @Positive
        @Override
    @Positive
        public List<Attribute.TypeCompound> getRawTypeAttributes();

    @Positive
        public Type erasure(Types types);

    @Positive
        public String className();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        @CanonicalName
    @Positive
        public Name getQualifiedName();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Symbol> getEnclosedElements();

    @Positive
        @BinaryName
    @Positive
        public Name flatName();

    @Positive
        public boolean isSubClass(Symbol base, Types types);

    @Positive
        public void complete() throws CompletionFailure;

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getInterfaces();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getSuperclass();

    @Positive
        @Override
    @Positive
        protected <A extends Annotation> A[] getInheritedAnnotations(Class<A> annoType);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Set<Modifier> getModifiers();

    @Positive
        public RecordComponent getRecordComponent(VarSymbol field);

    @Positive
        public RecordComponent getRecordComponent(JCVariableDecl var, boolean addIfMissing, List<JCAnnotation> annotations);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<? extends RecordComponent> getRecordComponents();

    @Positive
        public void setRecordComponents(List<RecordComponent> recordComponents);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public NestingKind getNestingKind();

    @Positive
        @Override
    @Positive
        protected <A extends Annotation> Attribute.Compound getAttribute(final Class<A> annoType);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
        public void markAbstractIfNeeded(Types types);

    @Positive
        public void reset();

    @Positive
        public void clearAnnotationMetadata();

    @Positive
        @Override
    @Positive
        public AnnotationTypeMetadata getAnnotationTypeMetadata();

    @Positive
        @Override
    @Positive
        public boolean isAnnotationType();

    @Positive
        public void setAnnotationTypeMetadata(AnnotationTypeMetadata a);

    @Positive
        public boolean isRecord();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getPermittedSubclasses();
    @Positive
    }

    @Positive
    public static class VarSymbol extends Symbol implements VariableElement {

    @Positive
        public int pos;

    @Positive
        public int adr;

    @Positive
        public VarSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        public MethodHandleSymbol asMethodHandle(boolean getter);

    @Positive
        public VarSymbol clone(Symbol newOwner);

    @Positive
        public String toString();

    @Positive
        public Symbol asMemberOf(Type site, Types types);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Object getConstantValue();

    @Positive
        public void setLazyConstValue(final Env<AttrContext> env, final Attr attr, final JCVariableDecl variable);

    @Positive
        public boolean isExceptionParameter();

    @Positive
        public boolean isResourceVariable();

    @Positive
        public Object getConstValue();

    @Positive
        public void setData(Object data);

    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class RecordComponent extends VarSymbol implements RecordComponentElement {

    @Positive
        public MethodSymbol accessor;

    @Positive
        public JCTree.JCMethodDecl accessorMeth;

    @Positive
        public RecordComponent(Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        public RecordComponent(VarSymbol field, List<JCAnnotation> annotations) {
    @Positive
        }

    @Positive
        public List<JCAnnotation> getOriginalAnnos();

    @Positive
        public boolean isVarargs();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ExecutableElement getAccessor();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ParamSymbol extends VarSymbol {

    @Positive
        public ParamSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Name getSimpleName();
    @Positive
    }

    @Positive
    public static class BindingSymbol extends VarSymbol {

    @Positive
        public BindingSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        public boolean isAliasFor(BindingSymbol b);

    @Positive
        List<BindingSymbol> aliases();

    @Positive
        public void preserveBinding();

    @Positive
        public boolean isPreserved();
    @Positive
    }

    @Positive
    public static class MethodSymbol extends Symbol implements ExecutableElement {

    @Positive
        public Code code;

    @Positive
        public List<VarSymbol> extraParams;

    @Positive
        public List<VarSymbol> capturedLocals;

    @Positive
        public List<VarSymbol> params;

    @Positive
        public Attribute defaultValue;

    @Positive
        public MethodSymbol(long flags, Name name, Type type, Symbol owner) {
    @Positive
        }

    @Positive
        public MethodSymbol clone(Symbol newOwner);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Set<Modifier> getModifiers();

    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        public boolean isHandle();

    @Positive
        public MethodHandleSymbol asHandle();

    @Positive
        public Symbol implemented(TypeSymbol c, Types types);

    @Positive
        public Symbol implementedIn(TypeSymbol c, Types types);

    @Positive
        public boolean binaryOverrides(Symbol _other, TypeSymbol origin, Types types);

    @Positive
        public MethodSymbol binaryImplementation(ClassSymbol origin, Types types);

    @Positive
        public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult);

    @Positive
        public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult, boolean requireConcreteIfInherited);

    @Positive
        @Override
    @Positive
        public boolean isInheritedIn(Symbol clazz, Types types);

    @Positive
        public boolean isLambdaMethod();

    @Positive
        public MethodSymbol originalEnclosingMethod();

    @Positive
        public MethodSymbol implementation(TypeSymbol origin, Types types, boolean checkResult);

    @Positive
        public static final Predicate<Symbol> implementation_filter;

    @Positive
        public MethodSymbol implementation(TypeSymbol origin, Types types, boolean checkResult, Predicate<Symbol> implFilter);

    @Positive
        public List<VarSymbol> params();

    @Positive
        public Symbol asMemberOf(Type site, Types types);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public ElementKind getKind();

    @Positive
        public boolean isStaticOrInstanceInit();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Attribute getDefaultValue();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<VarSymbol> getParameters();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean isVarArgs();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean isDefault();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getReceiverType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getReturnType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getThrownTypes();
    @Positive
    }

    @Positive
    public static class DynamicMethodSymbol extends MethodSymbol implements Dynamic {

    @Positive
        public LoadableConstant[] staticArgs;

    @Positive
        public MethodHandleSymbol bsm;

    @Positive
        public DynamicMethodSymbol(Name name, Symbol owner, MethodHandleSymbol bsm, Type type, LoadableConstant[] staticArgs) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public boolean isDynamic();

    @Positive
        @Override
    @Positive
        public LoadableConstant[] staticArgs();

    @Positive
        @Override
    @Positive
        public MethodHandleSymbol bootstrapMethod();

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        @Override
    @Positive
        public Type dynamicType();
    @Positive
    }

    @Positive
    public static class DynamicVarSymbol extends VarSymbol implements Dynamic, LoadableConstant {

    @Positive
        public LoadableConstant[] staticArgs;

    @Positive
        public MethodHandleSymbol bsm;

    @Positive
        public DynamicVarSymbol(Name name, Symbol owner, MethodHandleSymbol bsm, Type type, LoadableConstant[] staticArgs) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public boolean isDynamic();

    @Positive
        @Override
    @Positive
        public PoolConstant dynamicType();

    @Positive
        @Override
    @Positive
        public LoadableConstant[] staticArgs();

    @Positive
        @Override
    @Positive
        public LoadableConstant bootstrapMethod();

    @Positive
        @Override
    @Positive
        public int poolTag();
    @Positive
    }

    @Positive
    public static class MethodHandleSymbol extends MethodSymbol implements LoadableConstant {

    @Positive
        public MethodHandleSymbol(Symbol msym) {
    @Positive
        }

    @Positive
        public MethodHandleSymbol(Symbol msym, boolean getter) {
    @Positive
        }

    @Positive
        public int referenceKind();

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        @Override
    @Positive
        public Object poolKey(Types types);

    @Positive
        @Override
    @Positive
        public MethodHandleSymbol asHandle();

    @Positive
        @Override
    @Positive
        public Symbol baseSymbol();

    @Positive
        @Override
    @Positive
        public boolean isHandle();
    @Positive
    }

    @Positive
    public static class OperatorSymbol extends MethodSymbol {

    @Positive
        public int opcode;

    @Positive
        public OperatorSymbol(Name name, Type type, int opcode, Symbol owner) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @Positive
        public int getAccessCode(Tag tag);

    @Positive
        public enum AccessCode {

    @Positive
            UNKNOWN(-1, Tag.NO_TAG),
    @Positive
            DEREF(0, Tag.NO_TAG),
    @Positive
            ASSIGN(2, Tag.ASSIGN),
    @Positive
            PREINC(4, Tag.PREINC),
    @Positive
            PREDEC(6, Tag.PREDEC),
    @Positive
            POSTINC(8, Tag.POSTINC),
    @Positive
            POSTDEC(10, Tag.POSTDEC),
    @Positive
            FIRSTASGOP(12, Tag.NO_TAG);

    @Positive
            public final int code;

    @Positive
            public final Tag tag;

    @Positive
            public static final int numberOfAccessCodes;

    @Positive
            public static AccessCode getFromCode(int code);

    @Positive
            static int from(Tag tag, int opcode);
    @Positive
        }
    @Positive
    }

    @Positive
    @UsesObjectEquals
    @Positive
    public static interface Completer {

    @Positive
        @InternedDistinct
    @Positive
        public static final Completer NULL_COMPLETER;

    @Positive
        void complete(Symbol sym) throws CompletionFailure;

    @Positive
        default boolean isTerminal();
    @Positive
    }

    @Positive
    public static class CompletionFailure extends RuntimeException {

    @Positive
        public final transient DeferredCompletionFailureHandler dcfh;

    @Positive
        public transient Symbol sym;

    @Positive
        public CompletionFailure(Symbol sym, Supplier<JCDiagnostic> diagSupplier, DeferredCompletionFailureHandler dcfh) {
    @Positive
        }

    @Positive
        public JCDiagnostic getDiagnostic();

    @Positive
        @Override
    @Positive
        public String getMessage();

    @Positive
        public JCDiagnostic getDetailValue();

    @Positive
        @Override
    @Positive
        public CompletionFailure initCause(Throwable cause);

    @Positive
        public void resetDiagnostic(Supplier<JCDiagnostic> diagSupplier);
    @Positive
    }

    @Positive
    public interface Visitor<R, P> {

    @Positive
        R visitClassSymbol(ClassSymbol s, P arg);

    @Positive
        R visitMethodSymbol(MethodSymbol s, P arg);

    @Positive
        R visitPackageSymbol(PackageSymbol s, P arg);

    @Positive
        R visitOperatorSymbol(OperatorSymbol s, P arg);

    @Positive
        R visitVarSymbol(VarSymbol s, P arg);

    @Positive
        R visitTypeSymbol(TypeSymbol s, P arg);

    @Positive
        R visitSymbol(Symbol s, P arg);
    @Positive
    }
    @Positive
}
