/*
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.impl.Constants;
import com.sun.org.apache.xerces.internal.impl.RevalidationHandler;
import com.sun.org.apache.xerces.internal.impl.XMLEntityManager;
import com.sun.org.apache.xerces.internal.impl.XMLErrorReporter;
import com.sun.org.apache.xerces.internal.impl.dv.DatatypeException;
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
import com.sun.org.apache.xerces.internal.impl.dv.xs.XSSimpleTypeDecl;
import com.sun.org.apache.xerces.internal.impl.validation.ConfigurableValidationState;
import com.sun.org.apache.xerces.internal.impl.validation.ValidationManager;
import com.sun.org.apache.xerces.internal.impl.validation.ValidationState;
import com.sun.org.apache.xerces.internal.impl.xs.identity.Field;
import com.sun.org.apache.xerces.internal.impl.xs.identity.FieldActivator;
import com.sun.org.apache.xerces.internal.impl.xs.identity.IdentityConstraint;
import com.sun.org.apache.xerces.internal.impl.xs.identity.KeyRef;
import com.sun.org.apache.xerces.internal.impl.xs.identity.Selector;
import com.sun.org.apache.xerces.internal.impl.xs.identity.UniqueOrKey;
import com.sun.org.apache.xerces.internal.impl.xs.identity.ValueStore;
import com.sun.org.apache.xerces.internal.impl.xs.identity.XPathMatcher;
import com.sun.org.apache.xerces.internal.impl.xs.models.CMBuilder;
import com.sun.org.apache.xerces.internal.impl.xs.models.CMNodeFactory;
import com.sun.org.apache.xerces.internal.impl.xs.models.XSCMValidator;
import com.sun.org.apache.xerces.internal.impl.xs.util.XS10TypeHelper;
import com.sun.org.apache.xerces.internal.parsers.XMLParser;
import com.sun.org.apache.xerces.internal.util.AugmentationsImpl;
import com.sun.org.apache.xerces.internal.util.IntStack;
import com.sun.org.apache.xerces.internal.util.SymbolTable;
import com.sun.org.apache.xerces.internal.util.URI.MalformedURIException;
import com.sun.org.apache.xerces.internal.util.XMLAttributesImpl;
import com.sun.org.apache.xerces.internal.util.XMLChar;
import com.sun.org.apache.xerces.internal.util.XMLSymbols;
import com.sun.org.apache.xerces.internal.xni.Augmentations;
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
import com.sun.org.apache.xerces.internal.xni.QName;
import com.sun.org.apache.xerces.internal.xni.XMLAttributes;
import com.sun.org.apache.xerces.internal.xni.XMLDocumentHandler;
import com.sun.org.apache.xerces.internal.xni.XMLLocator;
import com.sun.org.apache.xerces.internal.xni.XMLResourceIdentifier;
import com.sun.org.apache.xerces.internal.xni.XMLString;
import com.sun.org.apache.xerces.internal.xni.XNIException;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;
import com.sun.org.apache.xerces.internal.xni.parser.XMLComponent;
import com.sun.org.apache.xerces.internal.xni.parser.XMLComponentManager;
import com.sun.org.apache.xerces.internal.xni.parser.XMLConfigurationException;
import com.sun.org.apache.xerces.internal.xni.parser.XMLDocumentFilter;
import com.sun.org.apache.xerces.internal.xni.parser.XMLDocumentSource;
import com.sun.org.apache.xerces.internal.xni.parser.XMLEntityResolver;
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
import com.sun.org.apache.xerces.internal.xs.AttributePSVI;
import com.sun.org.apache.xerces.internal.xs.ElementPSVI;
import com.sun.org.apache.xerces.internal.xs.ShortList;
import com.sun.org.apache.xerces.internal.xs.StringList;
import com.sun.org.apache.xerces.internal.xs.XSConstants;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.Vector;
import javax.xml.XMLConstants;
import jdk.xml.internal.JdkConstants;
import jdk.xml.internal.JdkXmlUtils;

public class XMLSchemaValidator implements XMLComponent, XMLDocumentFilter, FieldActivator, RevalidationHandler, XSElementDeclHelper {

    protected static final String VALIDATION;

    protected static final String SCHEMA_VALIDATION;

    protected static final String SCHEMA_FULL_CHECKING;

    protected static final String DYNAMIC_VALIDATION;

    protected static final String NORMALIZE_DATA;

    protected static final String SCHEMA_ELEMENT_DEFAULT;

    protected static final String SCHEMA_AUGMENT_PSVI;

    protected static final String ALLOW_JAVA_ENCODINGS;

    protected static final String STANDARD_URI_CONFORMANT_FEATURE;

    protected static final String GENERATE_SYNTHETIC_ANNOTATIONS;

    protected static final String VALIDATE_ANNOTATIONS;

    protected static final String HONOUR_ALL_SCHEMALOCATIONS;

    protected static final String USE_GRAMMAR_POOL_ONLY;

    protected static final String CONTINUE_AFTER_FATAL_ERROR;

    protected static final String PARSER_SETTINGS;

    protected static final String NAMESPACE_GROWTH;

    protected static final String TOLERATE_DUPLICATES;

    protected static final String IGNORE_XSI_TYPE;

    protected static final String ID_IDREF_CHECKING;

    protected static final String UNPARSED_ENTITY_CHECKING;

    protected static final String IDENTITY_CONSTRAINT_CHECKING;

    protected static final String REPORT_WHITESPACE;

    public static final String SYMBOL_TABLE;

    public static final String ERROR_REPORTER;

    public static final String ENTITY_RESOLVER;

    public static final String XMLGRAMMAR_POOL;

    protected static final String VALIDATION_MANAGER;

    protected static final String ENTITY_MANAGER;

    protected static final String SCHEMA_LOCATION;

    protected static final String SCHEMA_NONS_LOCATION;

    protected static final String JAXP_SCHEMA_SOURCE;

    protected static final String JAXP_SCHEMA_LANGUAGE;

    protected static final String ROOT_TYPE_DEF;

    protected static final String ROOT_ELEMENT_DECL;

    protected static final String SCHEMA_DV_FACTORY;

    protected static final String OVERRIDE_PARSER;

    protected static final String USE_CATALOG;

    protected static final int ID_CONSTRAINT_NUM;

    protected ElementPSVImpl fCurrentPSVI;

    protected final AugmentationsImpl fAugmentations;

    protected XMLString fDefaultValue;

    protected boolean fDynamicValidation;

    protected boolean fSchemaDynamicValidation;

    protected boolean fDoValidation;

    protected boolean fFullChecking;

    protected boolean fNormalizeData;

    protected boolean fSchemaElementDefault;

    protected boolean fAugPSVI;

    protected boolean fIdConstraint;

    protected boolean fUseGrammarPoolOnly;

    protected boolean fNamespaceGrowth;

    protected boolean fEntityRef;

    protected boolean fInCDATA;

    protected boolean fSawOnlyWhitespaceInElementContent;

    protected SymbolTable fSymbolTable;

    protected final class XSIErrorReporter {

        public void reset(XMLErrorReporter errorReporter);

        public void pushContext();

        public String[] popContext();

        public String[] mergeContext();

        public void reportError(String domain, String key, Object[] arguments, short severity) throws XNIException;

        public void reportError(XMLLocator location, String domain, String key, Object[] arguments, short severity) throws XNIException;
    }

    protected final XSIErrorReporter fXSIErrorReporter;

    protected XMLEntityResolver fEntityResolver;

    protected ValidationManager fValidationManager;

    protected ConfigurableValidationState fValidationState;

    protected XMLGrammarPool fGrammarPool;

    protected String fExternalSchemas;

    protected String fExternalNoNamespaceSchema;

    protected Object fJaxpSchemaSource;

    protected final XSDDescription fXSDDescription;

    protected final Map<String, XMLSchemaLoader.LocationArray> fLocationPairs;

    protected XMLDocumentHandler fDocumentHandler;

    protected XMLDocumentSource fDocumentSource;

    public String[] getRecognizedFeatures();

    public void setFeature(String featureId, boolean state) throws XMLConfigurationException;

    public String[] getRecognizedProperties();

    public void setProperty(String propertyId, Object value) throws XMLConfigurationException;

    public Boolean getFeatureDefault(String featureId);

    public Object getPropertyDefault(String propertyId);

    public void setDocumentHandler(XMLDocumentHandler documentHandler);

    public XMLDocumentHandler getDocumentHandler();

    public void setDocumentSource(XMLDocumentSource source);

    public XMLDocumentSource getDocumentSource();

    public void startDocument(XMLLocator locator, String encoding, NamespaceContext namespaceContext, Augmentations augs) throws XNIException;

    public void xmlDecl(String version, String encoding, String standalone, Augmentations augs) throws XNIException;

    public void doctypeDecl(String rootElement, String publicId, String systemId, Augmentations augs) throws XNIException;

    public void startElement(QName element, XMLAttributes attributes, Augmentations augs) throws XNIException;

    public void emptyElement(QName element, XMLAttributes attributes, Augmentations augs) throws XNIException;

    public void characters(XMLString text, Augmentations augs) throws XNIException;

    public void ignorableWhitespace(XMLString text, Augmentations augs) throws XNIException;

    public void endElement(QName element, Augmentations augs) throws XNIException;

    public void startCDATA(Augmentations augs) throws XNIException;

    public void endCDATA(Augmentations augs) throws XNIException;

    public void endDocument(Augmentations augs) throws XNIException;

    public boolean characterData(String data, Augmentations augs);

    public void elementDefault(String data);

    public void startGeneralEntity(String name, XMLResourceIdentifier identifier, String encoding, Augmentations augs) throws XNIException;

    public void textDecl(String version, String encoding, Augmentations augs) throws XNIException;

    public void comment(XMLString text, Augmentations augs) throws XNIException;

    public void processingInstruction(String target, XMLString data, Augmentations augs) throws XNIException;

    public void endGeneralEntity(String name, Augmentations augs) throws XNIException;

    protected XPathMatcherStack fMatcherStack;

    protected ValueStoreCache fValueStoreCache;

    public XMLSchemaValidator() {
    }

    public void reset(XMLComponentManager componentManager) throws XMLConfigurationException;

    public void startValueScopeFor(IdentityConstraint identityConstraint, int initialDepth);

    public XPathMatcher activateField(Field field, int initialDepth);

    public void endValueScopeFor(IdentityConstraint identityConstraint, int initialDepth);

    public XSElementDecl getGlobalElementDecl(QName element);

    void ensureStackCapacity();

    void handleStartDocument(XMLLocator locator, String encoding);

    void handleEndDocument();

    XMLString handleCharacters(XMLString text);

    void handleIgnorableWhitespace(XMLString text);

    Augmentations handleStartElement(QName element, XMLAttributes attributes, Augmentations augs);

    Augmentations handleEndElement(QName element, Augmentations augs);

    final Augmentations endElementPSVI(boolean root, SchemaGrammar[] grammars, Augmentations augs);

    Augmentations getEmptyAugs(Augmentations augs);

    void storeLocations(String sLocation, String nsLocation);

    SchemaGrammar findSchemaGrammar(short contextType, String namespace, QName enclosingElement, QName triggeringComponent, XMLAttributes attributes);

    XSTypeDefinition getAndCheckXsiType(QName element, String xsiType, XMLAttributes attributes);

    boolean getXsiNil(QName element, String xsiNil);

    void processAttributes(QName element, XMLAttributes attributes, XSAttributeGroupDecl attrGrp);

    void processOneAttribute(QName element, XMLAttributes attributes, int index, XSAttributeDecl currDecl, XSAttributeUseImpl currUse, AttributePSVImpl attrPSVI);

    void addDefaultAttributes(QName element, XMLAttributes attributes, XSAttributeGroupDecl attrGrp);

    void processElementContent(QName element);

    Object elementLocallyValidType(QName element, Object textContent);

    Object elementLocallyValidComplexType(QName element, Object textContent);

    void processRootTypeQName(final javax.xml.namespace.QName rootTypeQName);

    void processRootElementDeclQName(final javax.xml.namespace.QName rootElementDeclQName, final QName element);

    void checkElementMatchesRootElementDecl(final XSElementDecl rootElementDecl, final QName element);

    void reportSchemaError(String key, Object[] arguments);

    protected static class XPathMatcherStack {

        protected XPathMatcher[] fMatchers;

        protected int fMatchersCount;

        protected IntStack fContextStack;

        public XPathMatcherStack() {
        }

        public void clear();

        public int size();

        public int getMatcherCount();

        public void addMatcher(XPathMatcher matcher);

        public XPathMatcher getMatcherAt(int index);

        public void pushContext();

        public void popContext();
    }

    protected abstract class ValueStoreBase implements ValueStore {

        protected IdentityConstraint fIdentityConstraint;

        protected int fFieldCount;

        protected Field[] fFields;

        protected Object[] fLocalValues;

        protected short[] fLocalValueTypes;

        protected ShortList[] fLocalItemValueTypes;

        protected int fValuesCount;

        protected boolean fHasValue;

        public final Vector<Object> fValues;

        public ShortVector fValueTypes;

        public Vector<ShortList> fItemValueTypes;

        protected ValueStoreBase(IdentityConstraint identityConstraint) {
        }

        public void clear();

        public void append(ValueStoreBase newVal);

        public void startValueScope();

        public void endValueScope();

        public void endDocumentFragment();

        public void endDocument();

        public void reportError(String key, Object[] args);

        public void addValue(Field field, boolean mayMatch, Object actualValue, short valueType, ShortList itemValueType);

        @Pure
        public boolean contains();

        public int contains(ValueStoreBase vsb);

        protected void checkDuplicateValues();

        protected String toString(Object[] values);

        protected String toString(Vector<Object> values, int start, int length);

        public String toString();
    }

    protected class UniqueValueStore extends ValueStoreBase {

        public UniqueValueStore(UniqueOrKey unique) {
        }

        protected void checkDuplicateValues();
    }

    protected class KeyValueStore extends ValueStoreBase {

        public KeyValueStore(UniqueOrKey key) {
        }

        protected void checkDuplicateValues();
    }

    protected class KeyRefValueStore extends ValueStoreBase {

        protected ValueStoreBase fKeyValueStore;

        public KeyRefValueStore(KeyRef keyRef, KeyValueStore keyValueStore) {
        }

        public void endDocumentFragment();

        public void endDocument();
    }

    protected class ValueStoreCache {

        protected final List<ValueStoreBase> fValueStores;

        protected final Map<LocalIDKey, ValueStoreBase> fIdentityConstraint2ValueStoreMap;

        protected final Stack<Map<IdentityConstraint, ValueStoreBase>> fGlobalMapStack;

        protected final Map<IdentityConstraint, ValueStoreBase> fGlobalIDConstraintMap;

        public ValueStoreCache() {
        }

        public void startDocument();

        @SuppressWarnings("unchecked")
        public void startElement();

        public void endElement();

        public void initValueStoresFor(XSElementDecl eDecl, FieldActivator activator);

        public ValueStoreBase getValueStoreFor(IdentityConstraint id, int initialDepth);

        public ValueStoreBase getGlobalValueStoreFor(IdentityConstraint id);

        public void transplant(IdentityConstraint id, int initialDepth);

        public void endDocument();

        public String toString();
    }

    protected static final class LocalIDKey {

        public IdentityConstraint fId;

        public int fDepth;

        public LocalIDKey() {
        }

        public LocalIDKey(IdentityConstraint id, int depth) {
        }

        public int hashCode();

        public boolean equals(Object localIDKey);
    }

    protected static final class ShortVector {

        public ShortVector() {
        }

        public ShortVector(int initialCapacity) {
        }

        public int length();

        public void add(short value);

        public short valueAt(int position);

        public void clear();

        @Pure
        public boolean contains(short value);
    }
}
